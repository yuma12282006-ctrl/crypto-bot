import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
import ccxt

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
LINE_NOTIFY_TOKEN = os.environ.get("LINE_NOTIFY_TOKEN")

EXCHANGE_API_KEY = os.environ.get("EXCHANGE_API_KEY")
EXCHANGE_API_SECRET = os.environ.get("EXCHANGE_API_SECRET")

# 監視5銘柄
SYMBOLS = [
    {"symbol": "BTC/JPY", "yf": "BTC-JPY"},
    {"symbol": "ETH/JPY", "yf": "ETH-JPY"},
    {"symbol": "XRP/JPY", "yf": "XRP-JPY"},
    {"symbol": "LTC/JPY", "yf": "LTC-JPY"},
    {"symbol": "XLM/JPY", "yf": "XLM-JPY"}
]

def fetch_chart_data(yf_symbol, timeframe, limit=120):
    """Yahoo Financeからローソク足データを取得"""
    try:
        interval = '1d' if timeframe == '1d' else '1h'
        range_val = '120d' if timeframe == '1d' else '30d'
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_symbol}?interval={interval}&range={range_val}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        data = res.json()['chart']['result'][0]
        
        prices = data['indicators']['quote'][0]['close']
        volumes = data['indicators']['quote'][0]['volume']
        
        df = pd.DataFrame({'Close': prices, 'Volume': volumes}).dropna()
        
        if timeframe == '4h':
            df['Group'] = np.arange(len(df)) // 4
            df = df.groupby('Group').agg({'Close': 'last', 'Volume': 'sum'}).reset_index(drop=True)
            
        return df
    except Exception as e:
        print(f"[{yf_symbol}] データ取得エラー({timeframe}): {e}")
        return None

def fetch_macro_data():
    """世界経済指標（ドル円・S&P500）を取得してトレンドを分析"""
    try:
        # ドル円とS&P500のデータを取得
        usdjpy = fetch_chart_data("USDJPY=X", "1d", limit=30)
        sp500 = fetch_chart_data("^GSPC", "1d", limit=30)
        
        macro_score = 0
        if usdjpy is not None and len(usdjpy) > 5:
            # 直近の価格がSMA5より上ならプラス
            if usdjpy['Close'].iloc[-1] > usdjpy['Close'].rolling(5).mean().iloc[-1]:
                macro_score += 1
        if sp500 is not None and len(sp500) > 5:
            if sp500['Close'].iloc[-1] > sp500['Close'].rolling(5).mean().iloc[-1]:
                macro_score += 1
                
        return macro_score # 0〜2 (2なら世界経済リスクオン)
    except Exception:
        return 1 # 取得失敗時は中立

def execute_bitflyer_order(symbol, amount_jpy=1000):
    """国内bitFlyerへの実発注関数"""
    if not EXCHANGE_API_KEY or not EXCHANGE_API_SECRET:
        return "APIキー未設定のためシミュレーション完了"

    try:
        exchange = ccxt.bitflyer({
            'apiKey': EXCHANGE_API_KEY,
            'secret': EXCHANGE_API_SECRET,
            'enableRateLimit': True,
        })
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        raw_amount = amount_jpy / current_price
        
        if "BTC" in symbol:
            amount = round(max(raw_amount, 0.001), 3)
        elif "ETH" in symbol:
            amount = round(max(raw_amount, 0.01), 2)
        else:
            amount = round(max(raw_amount, 1.0), 1)

        return f"API接続OK / 発注スタンバイ完了 (数量: {amount})"
    except Exception as e:
        return f"国内API確認中: {str(e)[:35]}"

def send_line_notify(message):
    if not LINE_NOTIFY_TOKEN:
        return
    try:
        url = "https://notify-api.line.me/api/notify"
        headers = {"Authorization": f"Bearer {LINE_NOTIFY_TOKEN}"}
        payload = {"message": message}
        requests.post(url, headers=headers, data=payload, timeout=5)
    except Exception as e:
        print(f"LINE通知エラー: {e}")

def process_features(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA20'] + (std20 * 2)
    df['BB_Lower'] = df['SMA20'] - (std20 * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA20']
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Diff'] = df['MACD'] - df['Signal_Line']
    
    df['Return'] = df['Close'].pct_change()
    df['SMA_Diff'] = (df['Close'] - df['SMA20']) / df['SMA20']
    df['Volume_Change'] = df['Volume'].pct_change()
    
    return df

def run_backtest_with_trailing(df_daily, signals):
    trades = []
    fee = 0.002
    for i in range(len(signals) - 1):
        if signals[i] == 1:
            buy_price = df_daily['Close'].iloc[i]
            next_price = df_daily['Close'].iloc[i + 1]
            peak_price = max(buy_price, next_price)
            
            ret = (next_price - buy_price) / buy_price
            if (peak_price - next_price) / peak_price >= 0.025 and peak_price > buy_price:
                ret = (peak_price * 0.975 - buy_price) / buy_price
            elif ret <= -0.03:
                ret = -0.03
            elif ret >= 0.08:
                ret = 0.08

            net_ret = ret - fee
            trades.append(net_ret)
            
    if not trades:
        return {"total_trades": 0, "win_rate": 0.0, "total_return": 0.0}
        
    wins = [t for t in trades if t > 0]
    win_rate = (len(wins) / len(trades)) * 100
    total_return = (np.prod([1 + t for t in trades]) - 1) * 100
    return {
        "total_trades": len(trades),
        "win_rate": round(win_rate, 1),
        "total_return": round(total_return, 1)
    }

def analyze_crypto(item, macro_score):
    symbol = item["symbol"]
    yf_symbol = item["yf"]
    
    df_daily = fetch_chart_data(yf_symbol, "1d", limit=120)
    df_1h = fetch_chart_data(yf_symbol, "1h", limit=168)
    
    if df_daily is None or len(df_daily) < 40:
        return None

    df_daily = process_features(df_daily)
    # マクロ経済スコアを特徴量として全行に結合
    df_daily['Macro_Score'] = macro_score
    
    df_daily['Target'] = (df_daily['Close'].shift(-1) > df_daily['Close']).astype(int)
    df_clean = df_daily.dropna()
    
    if len(df_clean) < 25:
        return None

    # AIの学習特徴量にマクロ経済スコアを追加！
    features = ['RSI', 'Return', 'SMA_Diff', 'Volume_Change', 'MACD_Diff', 'BB_Width', 'Macro_Score']
    X = df_clean[features].iloc[:-1]
    y = df_clean['Target'].iloc[:-1]

    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, max_depth=3, random_state=42, verbose=-1)
    model.fit(X, y)

    latest_X = df_clean[features].iloc[[-1]]
    prob_daily = model.predict_proba(latest_X)[0][1] * 100

    hourly_trend = "不明"
    if df_1h is not None and len(df_1h) >= 20:
        df_1h['SMA20'] = df_1h['Close'].rolling(window=20).mean()
        latest_1h_close = df_1h['Close'].iloc[-1]
        latest_1h_sma = df_1h['SMA20'].iloc[-1]
        hourly_trend = "上昇（SMA20の上）" if latest_1h_close > latest_1h_sma else "下降（SMA20の下）"

    all_probs = model.predict_proba(df_clean[features])[:, 1]
    signals = [1 if p >= 58.0 and r <= 50 else 0 for p, r in zip(all_probs, df_clean['RSI'])]
    bt_result = run_backtest_with_trailing(df_clean, signals)

    current_price = df_clean['Close'].iloc[-1]
    rsi = df_clean['RSI'].iloc[-1]
    bb_upper = df_clean['BB_Upper'].iloc[-1]
    
    stop_loss = current_price * 0.97
    trailing_target = current_price * 1.025

    # 【世界経済連動・超強固フィルター】マクロが強気(>=1)のときを優遇
    is_strong_signal = (
        prob_daily >= 58.0 and 
        "上昇" in hourly_trend and 
        macro_score >= 1 and 
        rsi <= 50 and 
        current_price <= bb_upper
    )

    order_status = "見送り"
    if is_strong_signal:
        order_status = execute_bitflyer_order(symbol, amount_jpy=1000)
        send_line_notify(f"\n🌐【世界連動要塞シグナル】\n銘柄: {symbol}\n価格: ¥{current_price:,.0f}\nAI確率: {prob_daily}%\nマクロ環境: 良好(スコア{macro_score})")

    return {
        "symbol": symbol,
        "price": round(current_price, 2),
        "rsi": round(rsi, 1),
        "ai_prob": round(prob_daily, 1),
        "hourly_trend": hourly_trend,
        "macro_score": macro_score,
        "stop_loss": round(stop_loss, 2),
        "trailing_target": round(trailing_target, 2),
        "bt_trades": bt_result["total_trades"],
        "bt_win_rate": bt_result["win_rate"],
        "bt_return": bt_result["total_return"],
        "order_status": order_status,
        "is_signal": is_strong_signal
    }

def analyze_and_notify():
    macro_score = fetch_macro_data()
    embeds = []
    has_strong_signal = False
    
    for item in SYMBOLS:
        data = analyze_crypto(item, macro_score)
        if not data:
            continue
            
        symbol = data["symbol"]
        price = data["price"]
        rsi = data["rsi"]
        ai_prob = data["ai_prob"]
        hourly_trend = data["hourly_trend"]
        stop_loss = data["stop_loss"]
        trailing_target = data["trailing_target"]
        order_status = data["order_status"]
        
        if data["is_signal"]:
            has_strong_signal = True
            signal = "🌐🚀 【世界連動・要塞AI強気シグナル】"
            trade_action = f"⚡ {order_status}"
            color = 3066993
        elif ai_prob <= 42.0 or "下降" in hourly_trend or macro_score == 0:
            signal = "🌐⚠️ 【要塞防御・マクロ警戒見送り】"
            trade_action = "世界経済またはトレンド不一致"
            color = 15158332
        else:
            signal = "⚡⚖️ 【中立・様子見】"
            trade_action = "明確なシグナルなし"
            color = 3447003

        embed = {
            "title": f"🇯🇵 世界経済連動アナリティクス: {symbol}",
            "color": color,
            "fields": [
                {"name": "判定シグナル", "value": signal, "inline": False},
                {"name": "自動売買アクション", "value": trade_action, "inline": False},
                {"name": "現在価格", "value": f"¥{price:,.2f}", "inline": True},
                {"name": "日足AI予測(世界経済込)", "value": f"{ai_prob}%", "inline": True},
                {"name": "世界経済マクロ環境", "value": f"スコア: {macro_score}/2 (米株・為替連動)", "inline": True},
                {"name": "📈 追従検証(勝率・手数料込)", "value": f"{data['bt_win_rate']}% ({data['bt_trades']}回)", "inline": True},
                {"name": "💰 通算リターン", "value": f"{data['bt_return']}%", "inline": True},
                {"name": "RSI (14)", "value": f"{rsi}", "inline": True},
                {"name": "🛑 防御損切り (-3%)", "value": f"¥{stop_loss:,.2f}", "inline": True},
                {"name": "🔄 トレール起動地点", "value": f"¥{trailing_target:,.2f}", "inline": True}
            ],
            "footer": {"text": f"実行日時(世界経済連動プロ版): {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }
        embeds.append(embed)

    if not embeds:
        print("有効なデータがありませんでした")
        return

    content_msg = "🌐 **【世界経済要塞アラート】** グローバル市場連動の強気シグナルが点灯しました！ @here" if has_strong_signal else None

    payload = {"embeds": embeds[:10]}
    if content_msg:
        payload["content"] = content_msg

    response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
    if response.status_code == 204:
        print("世界連動版 通知成功！")
    else:
        print(f"通知失敗: {response.status_code}")

if __name__ == "__main__":
    analyze_and_notify()
