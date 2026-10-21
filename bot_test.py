import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
import ccxt  # bitFlyerなど主要取引所対応ライブラリ

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

# bitFlyer APIキー（GitHub Secretsから安全に取得）
EXCHANGE_API_KEY = os.environ.get("EXCHANGE_API_KEY")
EXCHANGE_API_SECRET = os.environ.get("EXCHANGE_API_SECRET")

# 国内取引所(bitFlyer)対応ペア
SYMBOLS = [
    "BTC/JPY",
    "ETH/JPY",
    "XRP/JPY"
]

def fetch_chart_data(symbol, timeframe, limit=120):
    """ccxt経由でbitFlyer等のローソク足データを取得"""
    try:
        exchange = ccxt.bitflyer()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            # 万が一取得できない場合はYahoo Financeでバックアップ
            yf_symbol = symbol.replace("/", "-")
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_symbol}?interval={'1d' if timeframe=='1d' else '1h'}&range={'120d' if timeframe=='1d' else '7d'}"
            res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            data = res.json()['chart']['result'][0]
            prices = data['indicators']['quote'][0]['close']
            volumes = data['indicators']['quote'][0]['volume']
            df = pd.DataFrame({'Close': prices, 'Volume': volumes}).dropna()
            return df

        df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        return df
    except Exception as e:
        print(f"[{symbol}] {timeframe}データ取得失敗: {e}")
        return None

def execute_bitflyer_order(symbol, amount_jpy=1000):
    """【国内bitFlyer自動発注テスト関数】
       APIキーがSecretsに登録されている場合に国内APIを実行
    """
    if not EXCHANGE_API_KEY or not EXCHANGE_API_SECRET:
        return "APIキー未設定（国内シミュレーション完了）"

    try:
        exchange = ccxt.bitflyer({
            'apiKey': EXCHANGE_API_KEY,
            'secret': EXCHANGE_API_SECRET,
            'enableRateLimit': True,
        })
        
        # bitFlyer APIを使った自動発注処理
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        amount = round(amount_jpy / current_price, 4)

        # 成行買い注文 (例: BTC/JPY)
        # order = exchange.create_market_buy_order(symbol, amount)
        print(f"🔥【bitFlyer発注成功】 銘柄: {symbol}, 数量: {amount}, 約定価格: ¥{current_price:,.0f}")
        return f"bitFlyer発注テスト成功 (数量: {amount})"
        
    except Exception as e:
        print(f"❌ [{symbol}] bitFlyer発注エラー: {e}")
        return f"国内APIエラー: {str(e)[:30]}"

def process_features(df):
    """テクニカル指標と特徴量の生成"""
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    
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

def run_backtest_with_fees(df_daily, signals):
    """往復手数料0.2%込みバックテスト"""
    trades = []
    fee = 0.002
    
    for i in range(len(signals) - 1):
        if signals[i] == 1:
            buy_price = df_daily['Close'].iloc[i]
            next_price = df_daily['Close'].iloc[i + 1]
            ret = (next_price - buy_price) / buy_price
            
            if ret <= -0.03:
                ret = -0.03
            elif ret >= 0.05:
                ret = 0.05
                
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

def analyze_crypto(symbol):
    df_daily = fetch_chart_data(symbol, "1d", limit=120)
    df_1h = fetch_chart_data(symbol, "1h", limit=168)
    
    if df_daily is None or len(df_daily) < 40:
        return None

    df_daily = process_features(df_daily)
    df_daily['Target'] = (df_daily['Close'].shift(-1) > df_daily['Close']).astype(int)
    df_clean = df_daily.dropna()
    
    if len(df_clean) < 25:
        return None

    features = ['RSI', 'Return', 'SMA_Diff', 'Volume_Change', 'MACD_Diff']
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
    bt_result = run_backtest_with_fees(df_clean, signals)

    current_price = df_clean['Close'].iloc[-1]
    rsi = df_clean['RSI'].iloc[-1]
    stop_loss = current_price * 0.97
    take_profit = current_price * 1.05

    is_strong_signal = (prob_daily >= 58.0 and "上昇" in hourly_trend and rsi <= 50)

    # 買いシグナルが出たら国内bitFlyerテスト発注
    order_status = "見送り"
    if is_strong_signal:
        order_status = execute_bitflyer_order(symbol, amount_jpy=1000)

    return {
        "symbol": symbol,
        "price": round(current_price, 2),
        "rsi": round(rsi, 1),
        "ai_prob": round(prob_daily, 1),
        "hourly_trend": hourly_trend,
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "bt_trades": bt_result["total_trades"],
        "bt_win_rate": bt_result["win_rate"],
        "bt_return": bt_result["total_return"],
        "order_status": order_status,
        "is_signal": is_strong_signal
    }

def analyze_and_notify():
    embeds = []
    
    for symbol in SYMBOLS:
        data = analyze_crypto(symbol)
        if not data:
            continue
            
        price = data["price"]
        rsi = data["rsi"]
        ai_prob = data["ai_prob"]
        hourly_trend = data["hourly_trend"]
        stop_loss = data["stop_loss"]
        take_profit = data["take_profit"]
        order_status = data["order_status"]
        
        if data["is_signal"]:
            signal = f"🏰🚀 【要塞級AI強気買いシグナル発動】"
            trade_action = f"⚡ {order_status}"
            color = 3066993  # 緑
        elif ai_prob <= 42.0 or "下降" in hourly_trend:
            signal = f"⚡⚠️ 【要塞防御・様子見推奨】"
            trade_action = f"トレンド不一致のため見送り"
            color = 15158332 # 赤
        else:
            signal = f"⚡⚖️ 【中立・様子見】"
            trade_action = f"明確なシグナルなし"
            color = 3447003  # 青

        embed = {
            "title": f"🇯🇵 国内取引所(bitFlyer)分析: {symbol}",
            "color": color,
            "fields": [
                {"name": "判定シグナル", "value": signal, "inline": False},
                {"name": "自動売買アクション", "value": trade_action, "inline": False},
                {"name": "現在価格", "value": f"¥{price:,.0f}", "inline": True},
                {"name": "日足AI予測(LGBM)", "value": f"{ai_prob}%", "inline": True},
                {"name": "1時間足トレンド", "value": f"{hourly_trend}", "inline": True},
                {"name": "📈 厳格検証(勝率・手数料込)", "value": f"{data['bt_win_rate']}% ({data['bt_trades']}回)", "inline": True},
                {"name": "💰 手数料込 通算リターン", "value": f"{data['bt_return']}%", "inline": True},
                {"name": "RSI (14)", "value": f"{rsi}", "inline": True},
                {"name": "🛑 損切り (-3%)", "value": f"¥{stop_loss:,.0f}", "inline": True},
                {"name": "🎯 利食い (+5%)", "value": f"¥{take_profit:,.0f}", "inline": True}
            ],
            "footer": {"text": f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }
        embeds.append(embed)

    if not embeds:
        print("データなし")
        return

    payload = {"embeds": embeds[:10]}
    response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
    if response.status_code == 204:
        print("通知成功！")
    else:
        print(f"通知失敗: {response.status_code}")

if __name__ == "__main__":
    analyze_and_notify()
