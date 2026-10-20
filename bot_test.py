import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

SYMBOLS = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "DOGE-USD",
    "BNB-USD",
    "ADA-USD",
    "AVAX-USD"
]

def fetch_chart_data(symbol, interval, range_val):
    """Yahoo Finance APIから指定した足のデータを取得"""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_val}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return None
        data = res.json()['chart']['result'][0]
        prices = data['indicators']['quote'][0]['close']
        volumes = data['indicators']['quote'][0]['volume']
        df = pd.DataFrame({'Close': prices, 'Volume': volumes}).dropna()
        return df
    except Exception as e:
        print(f"[{symbol}] {interval}データ取得失敗: {e}")
        return None

def process_features(df):
    """テクニカル指標と特徴量の生成"""
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    
    # RSI(14)
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
    
    # 変化率
    df['Return'] = df['Close'].pct_change()
    df['SMA_Diff'] = (df['Close'] - df['SMA20']) / df['SMA20']
    df['Volume_Change'] = df['Volume'].pct_change()
    
    return df

def run_backtest(df_daily, signals):
    """過去データのバックテスト計算（シグナルが出たら翌日買い、損切り-3%/利食い+5%または1日保有）"""
    trades = []
    
    for i in range(len(signals) - 1):
        if signals[i] == 1:  # 買いシグナル
            buy_price = df_daily['Close'].iloc[i]
            next_price = df_daily['Close'].iloc[i + 1]
            ret = (next_price - buy_price) / buy_price
            
            # リスク管理ルール適用
            if ret <= -0.03:
                ret = -0.03
            elif ret >= 0.05:
                ret = 0.05
                
            trades.append(ret)
            
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
    """マルチタイムフレーム分析 ＋ LightGBM ＋ バックテスト"""
    # 1. 日足データ (長期) & 1時間足データ (短期) の取得
    df_daily = fetch_chart_data(symbol, "1d", "120d")
    df_1h = fetch_chart_data(symbol, "1h", "7d")
    
    if df_daily is None or len(df_daily) < 40:
        return None

    # 特徴量計算
    df_daily = process_features(df_daily)
    
    # 目的変数: 翌日終値が上昇なら1
    df_daily['Target'] = (df_daily['Close'].shift(-1) > df_daily['Close']).astype(int)
    df_clean = df_daily.dropna()
    
    if len(df_clean) < 25:
        return None

    features = ['RSI', 'Return', 'SMA_Diff', 'Volume_Change', 'MACD_Diff']
    X = df_clean[features].iloc[:-1]
    y = df_clean['Target'].iloc[:-1]

    # LightGBMモデルの学習
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, max_depth=3, random_state=42, verbose=-1)
    model.fit(X, y)

    # 日足でのAI予測値
    latest_X = df_clean[features].iloc[[-1]]
    prob_daily = model.predict_proba(latest_X)[0][1] * 100

    # 1時間足の短期トレンド判定
    hourly_trend = "不明"
    if df_1h is not None and len(df_1h) >= 20:
        df_1h['SMA20'] = df_1h['Close'].rolling(window=20).mean()
        latest_1h_close = df_1h['Close'].iloc[-1]
        latest_1h_sma = df_1h['SMA20'].iloc[-1]
        hourly_trend = "上昇（SMA20の上）" if latest_1h_close > latest_1h_sma else "下降（SMA20の下）"

    # バックテスト実行 (過去シグナルの生成)
    all_probs = model.predict_proba(df_clean[features])[:, 1]
    signals = [1 if p >= 0.58 and r <= 50 else 0 for p, r in zip(all_probs, df_clean['RSI'])]
    bt_result = run_backtest(df_clean, signals)

    current_price = df_clean['Close'].iloc[-1]
    rsi = df_clean['RSI'].iloc[-1]
    stop_loss = current_price * 0.97
    take_profit = current_price * 1.05

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
        "bt_return": bt_result["total_return"]
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
        
        # マルチタイムフレーム判定
        if ai_prob >= 58.0 and "上昇" in hourly_trend and rsi <= 50:
            signal = f"🔥🚀 【マルチタイムフレーム最上位買いシグナル】"
            trade_action = f"日足AI＆1時間足トレンド合致！注文推奨"
            color = 3066993  # 緑
        elif ai_prob <= 42.0 or "下降" in hourly_trend:
            signal = f"⚡⚠️ 【売り警戒・短期下降トレンド】"
            trade_action = f"ポジション調整・様子見"
            color = 15158332 # 赤
        else:
            signal = f"⚡⚖️ 【様子見・シグナル待ち】"
            trade_action = f"明確な根拠なし"
            color = 3447003  # 青

        embed = {
            "title": f"📊 マルチタイムAI ＆ バックテスト: {symbol}",
            "color": color,
            "fields": [
                {"name": "判定シグナル", "value": signal, "inline": False},
                {"name": "自動売買アクション", "value": trade_action, "inline": False},
                {"name": "現在価格", "value": f"${price:,.2f}", "inline": True},
                {"name": "日足AI予測(LGBM)", "value": f"{ai_prob}%", "inline": True},
                {"name": "1時間足トレンド", "value": f"{hourly_trend}", "inline": True},
                {"name": "📈 過去検証(勝率)", "value": f"{data['bt_win_rate']}% ({data['bt_trades']}回)", "inline": True},
                {"name": "💰 過去通算リターン", "value": f"{data['bt_return']}%", "inline": True},
                {"name": "RSI (14)", "value": f"{rsi}", "inline": True},
                {"name": "🛑 損切り (-3%)", "value": f"${stop_loss:,.2f}", "inline": True},
                {"name": "🎯 利食い (+5%)", "value": f"${take_profit:,.2f}", "inline": True}
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
