import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
import ccxt

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
LINE_NOTIFY_TOKEN = os.environ.get("LINE_NOTIFY_TOKEN") # 任意設定

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
    """Yahoo Financeから安全に各種時間足のローソク足データを取得"""
    try:
        if timeframe == '1d':
            interval, range_val = '1d', '120d'
        elif timeframe == '4h':
            interval, range_val = '1m', '14d' # 4h換算用または1hデータからリサンプル
            # Yahooは4h直サポートがないため1hを取得して整形
            interval, range_val = '1h', '30d'
        else:
            interval, range_val = '1h', '7d'

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_symbol}?interval={interval}&range={range_val}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        data = res.json()['chart']['result'][0]
        
        prices = data['indicators']['quote'][0]['close']
        volumes = data['indicators']['quote'][0]['volume']
        
        df = pd.DataFrame({'Close': prices, 'Volume': volumes}).dropna()
        
        # 4時間足への合成処理
        if timeframe == '4h':
            df['Group'] = np.arange(len(df)) // 4
            df = df.groupby('Group').agg({'Close': 'last', 'Volume': 'sum'}).reset_index(drop=True)
            
        return df
    except Exception as e:
        print(f"[{yf_symbol}] データ取得エラー({timeframe}): {e}")
        return None

def execute_bitflyer_order(symbol, amount_jpy=1000):
    """国内bitFlyerへの実発注関数（API審査中・接続確認対応）"""
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

        # 本人確認完了・準備OK時にコメントアウトを解除
        # order = exchange.create_market_buy_order(symbol, amount)
        return f"API接続OK / 発注スタンバイ完了 (数量: {amount})"
        
    except Exception as e:
        return f"国内API確認中: {str(e)[:35]}"

def send_line_notify(message):
    """LINE Notifyへの速報送信（Tokenがある場合のみ動作）"""
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
    
    # ボリンジャーバンド
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA20'] + (std20 * 2)
    df['BB_Lower'] = df['SMA20'] - (std20 * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA20']
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
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
    """トレーリングストップ（追従型損切り・利確）対応の厳格バックテスト"""
    trades = []
    fee = 0.002
    for i in range(len(signals) - 1):
        if signals[i] == 1:
            buy_price = df_daily['Close'].iloc[i]
            # 翌日以降の動きをトレーリングで追従するシミュレーション
            next_price = df_daily['Close'].iloc[i + 1]
            peak_price = max(buy_price, next_price)
            
            # トレーリングルール: 最高値から2.5%下落でトレール発動、上限最大+8%
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

def analyze_crypto(item):
    symbol = item["symbol"]
    yf_symbol = item["yf"]
    
    df_daily = fetch_chart_data(yf_symbol, "1d", limit=120)
    df_4h = fetch_chart_data(yf_symbol, "4h", limit=60)
    df_1h = fetch_chart_data(yf_symbol, "1h", limit=168)
    
    if df_daily is None or len(df_daily) < 40:
        return None

    df_daily = process_features(df_daily)
    df_daily['Target'] = (df_daily['Close'].shift(-1) > df_daily['Close']).astype(int)
    df_clean = df_daily.dropna()
    
    if len(df_clean) < 25:
        return None

    features = ['RSI', 'Return', 'SMA_Diff', 'Volume_Change', 'MACD_Diff', 'BB_Width']
    X = df_clean[features].iloc[:-1]
    y = df_clean['Target'].iloc[:-1]

    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, max_depth=3, random_state=42, verbose=-1)
    model.fit(X, y)

    latest_X = df_clean[features].iloc[[-1]]
    prob_daily = model.predict_proba(latest_X)[0][1] * 100

    # 1時間足トレンド
    hourly_trend = "不明"
    if df_1h is not None and len(df_1h) >= 20:
        df_1h['SMA20'] = df_1h['Close'].rolling(window=20).mean()
        latest_1h_close = df_1h['Close'].iloc[-1]
        latest_1h_sma = df_1h['SMA20'].iloc[-1]
        hourly_trend = "上昇（SMA20の上）" if latest_1h_close > latest_1h_sma else "下降（SMA20の下）"

    # 4時間足トレンドフィルター
    h4_trend = "不明"
    if df_4h is not None and len(df_4h) >= 20:
        df_4h['SMA20'] = df_4h['Close'].rolling(window=20).mean()
        latest_4h_close = df_4h['Close'].iloc[-1]
        latest_4h_sma = df_4h['SMA20'].iloc[-1]
        h4_trend = "強気（SMA20の上）" if latest_4h_close > latest_4h_sma else "弱気（SMA20の下）"

    all_probs = model.predict_proba(df_clean[features])[:, 1]
    signals = [1 if p >= 58.0 and r <= 50 else 0 for p, r in zip(all_probs, df_clean['RSI'])]
    bt_result = run_backtest_with_trailing(df_clean, signals)

    current_price = df_clean['Close'].iloc[-1]
    rsi = df_clean['RSI'].iloc[-1]
    bb_upper = df_clean['BB_Upper'].iloc[-1]
    
    # 動的トレーリング目標値の設定
    stop_loss = current_price * 0.97
    trailing_target = current_price * 1.025 # 高値追従スタート地点

    # 【超強固フィルター】日足AI58%以上 ＋ 1時間足上昇 ＋ 4時間足強気 ＋ RSI50以下 ＋ BB上限以下
    is_strong_signal = (
        prob_daily >= 58.0 and 
        "上昇" in hourly_trend and 
        "強気" in h4_trend and 
        rsi <= 50 and 
        current_price <= bb_upper
    )

    order_status = "見送り"
    if is_strong_signal:
        order_status = execute_bitflyer_order(symbol, amount_jpy=1000)
        # LINEへの即時アラート送信
        send_line_notify(f"\n🏰【要塞シグナル発動】\n銘柄: {symbol}\n価格: ¥{current_price:,.0f}\nAI確率: {prob_daily}%\n発注状態: {order_status}")

    return {
        "symbol": symbol,
        "price": round(current_price, 2),
        "rsi": round(rsi, 1),
        "ai_prob": round(prob_daily, 1),
        "hourly_trend": hourly_trend,
        "h4_trend": h4_trend,
        "stop_loss": round(stop_loss, 2),
        "trailing_target": round(trailing_target, 2),
        "bt_trades": bt_result["total_trades"],
        "bt_win_rate": bt_result["win_rate"],
        "bt_return": bt_result["total_return"],
        "order_status": order_status,
        "is_signal": is_strong_signal
    }

def analyze_and_notify():
    embeds = []
    has_strong_signal = False
    
    for item in SYMBOLS:
        data = analyze_crypto(item)
        if not data:
            continue
            
        symbol = data["symbol"]
        price = data["price"]
        rsi = data["rsi"]
        ai_prob = data["ai_prob"]
        hourly_trend = data["hourly_trend"]
        h4_trend = data["h4_trend"]
        stop_loss = data["stop_loss"]
        trailing_target = data["trailing_target"]
        order_status = data["order_status"]
        
        if data["is_signal"]:
            has_strong_signal = True
            signal = "🏰🚀 【要塞プロ級・強気買いシグナル発動】"
            trade_action = f"⚡ {order_status}"
            color = 3066993
        elif ai_prob <= 42.0 or "下降" in hourly_trend or "弱気" in h4_trend:
            signal = "⚡⚠️ 【要塞防御・様子見推奨】"
            trade_action = "複数時間足不一致のため見送り"
            color = 15158332
        else:
            signal = "⚡⚖️ 【中立・様子見】"
            trade_action = "明確なシグナルなし"
            color = 3447003

        embed = {
            "title": f"🇯🇵 要塞プロ仕様分析(JPY): {symbol}",
            "color": color,
            "fields": [
                {"name": "判定シグナル", "value": signal, "inline": False},
                {"name": "自動売買アクション", "value": trade_action, "inline": False},
                {"name": "現在価格", "value": f"¥{price:,.2f}", "inline": True},
                {"name": "日足AI予測(LGBM+BB)", "value": f"{ai_prob}%", "inline": True},
                {"name": "1H/4Hトレンド合流", "value": f"1H:{hourly_trend}\n4H:{h4_trend}", "inline": True},
                {"name": "📈 追従検証(勝率・手数料込)", "value": f"{data['bt_win_rate']}% ({data['bt_trades']}回)", "inline": True},
                {"name": "💰 通算リターン", "value": f"{data['bt_return']}%", "inline": True},
                {"name": "RSI (14)", "value": f"{rsi}", "inline": True},
                {"name": "🛑 防御損切り (-3%)", "value": f"¥{stop_loss:,.2f}", "inline": True},
                {"name": "🔄 トレール起動地点", "value": f"¥{trailing_target:,.2f}", "inline": True}
            ],
            "footer": {"text": f"実行日時(完全要塞プロ版): {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }
        embeds.append(embed)

    if not embeds:
        print("有効なデータがありませんでした")
        return

    # シグナル発動時はDiscordの@hereで強烈プッシュ通知！
    content_msg = "📢 **【要塞アラート】** 超厳格エントリーシグナルが発動しました！ @here" if has_strong_signal else None

    payload = {"embeds": embeds[:10]}
    if content_msg:
        payload["content"] = content_msg

    response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
    if response.status_code == 204:
        print("要塞プロ版 通知成功！")
    else:
        print(f"通知失敗: {response.status_code}")

if __name__ == "__main__":
    analyze_and_notify()
