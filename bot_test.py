import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb

# GitHub Secretsから安全にDiscordのWebhook URLを読み込む
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

# 監視銘柄
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

def get_crypto_data(symbol):
    """Yahoo Financeからデータを取得し、LightGBMと高度な指標で翌日の上昇確率を予測する"""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=120d"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"[{symbol}] データ取得失敗 (Status Code: {response.status_code})")
            return None

        data = response.json()
        result = data['chart']['result'][0]
        
        prices = result['indicators']['quote'][0]['close']
        highs = result['indicators']['quote'][0]['high']
        lows = result['indicators']['quote'][0]['low']
        volumes = result['indicators']['quote'][0]['volume']
        
        df = pd.DataFrame({
            'Close': prices,
            'High': highs,
            'Low': lows,
            'Volume': volumes
        })
        df = df.dropna()
        
        if len(df) < 50:
            print(f"[{symbol}] データ件数が不足しています")
            return None

        # 1. テクニカル指標の計算 (SMA20, RSI, MACD)
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

        # 2. ボリンジャーバンド（バンド幅の広がりを測定）
        std20 = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['SMA20'] + (std20 * 2)
        df['BB_Lower'] = df['SMA20'] - (std20 * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA20']

        # 3. ATR (Average True Range - 相場のボラティリティ・荒れ具合)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean() / df['Close']

        # 4. 特徴量のまとめ
        df['Return'] = df['Close'].pct_change()
        df['SMA_Diff'] = (df['Close'] - df['SMA20']) / df['SMA20']
        df['Volume_Change'] = df['Volume'].pct_change()
        df['MACD_Diff'] = df['MACD'] - df['Signal_Line']
        
        # 正解ラベル（翌日の終値が上がっていれば1、下がっていれば0）
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        df = df.dropna()
        
        if len(df) < 25:
            return None

        # 特徴量リスト
        features = ['RSI', 'Return', 'SMA_Diff', 'Volume_Change', 'MACD_Diff', 'BB_Width', 'ATR']
        X = df[features].iloc[:-1]
        y = df['Target'].iloc[:-1]

        # 最強AIモデル「LightGBM」による学習
        model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, max_depth=3, random_state=42, verbose=-1)
        model.fit(X, y)

        latest_X = df[features].iloc[[-1]]
        prob_up = model.predict_proba(latest_X)[0][1] * 100

        current_price = df['Close'].iloc[-1]
        sma20 = df['SMA20'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        atr = df['ATR'].iloc[-1] * 100  # %表示
        
        # リスク管理ラインの計算（損切り -3%、利食い +5%）
        stop_loss = current_price * 0.97
        take_profit = current_price * 1.05

        return {
            "symbol": symbol,
            "price": round(current_price, 2),
            "sma20": round(sma20, 2),
            "rsi": round(rsi, 1),
            "ai_prob": round(prob_up, 1),
            "atr": round(atr, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2)
        }
        
    except Exception as e:
        print(f"[{symbol}] エラー発生: {e}")
        return None

def analyze_and_notify():
    """データを解析し、LightGBMの高度な予測とリスク管理をDiscordに通知する"""
    embeds = []
    
    for symbol in SYMBOLS:
        data = get_crypto_data(symbol)
        if not data:
            continue
            
        price = data["price"]
        sma20 = data["sma20"]
        rsi = data["rsi"]
        ai_prob = data["ai_prob"]
        atr = data["atr"]
        stop_loss = data["stop_loss"]
        take_profit = data["take_profit"]
        
        # 判定ロジック（LightGBMの予測精度を活かした高度な判定）
        if ai_prob >= 60.0 and rsi <= 45:
            signal = f"⚡💎 【LightGBM 超強気買いシグナル】"
            trade_action = f"最高精度AIによる仮想買い注文実行！"
            color = 3066993  # 緑色
        elif ai_prob <= 40.0 or rsi >= 70:
            signal = f"⚡⚠️ 【LightGBM 売り警戒・様子見】"
            trade_action = f"ポジション調整・様子見"
            color = 15158332 # 赤色
        else:
            signal = f"⚡⚖️ 【LightGBM 中立・様子見】"
            trade_action = f"明確なシグナルなし"
            color = 3447003  # 青色

        embed = {
            "title": f"⚡ 究極AI判定 (LightGBM): {symbol}",
            "color": color,
            "fields": [
                {"name": "判定シグナル", "value": signal, "inline": False},
                {"name": "自動売買アクション", "value": trade_action, "inline": False},
                {"name": "現在価格", "value": f"${price:,.2f}", "inline": True},
                {"name": "AI上昇予測(LGBM)", "value": f"{ai_prob}%", "inline": True},
                {"name": "RSI (14)", "value": f"{rsi}", "inline": True},
                {"name": "市場ボラティリティ (ATR)", "value": f"{atr}%", "inline": True},
                {"name": "🛑 損切りライン (-3%)", "value": f"${stop_loss:,.2f}", "inline": True},
                {"name": "🎯 利食いライン (+5%)", "value": f"${take_profit:,.2f}", "inline": True}
            ],
            "footer": {"text": f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }
        embeds.append(embed)

    if not embeds:
        print("通知するデータがありませんでした。")
        return

    payload = {"embeds": embeds[:10]}
    response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
    
    if response.status_code == 204:
        print("Discord通知が成功しました！")
    else:
        print(f"Discord通知に失敗しました: {response.status_code}, {response.text}")

if __name__ == "__main__":
    analyze_and_notify()
