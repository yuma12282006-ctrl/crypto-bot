import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression

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
    """Yahoo Financeからデータを取得し、AIで翌日の上昇確率を予測する"""
    try:
        # 機械学習の学習用に少し長めのデータ（60日分）を取得
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=60d"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"[{symbol}] データ取得失敗 (Status Code: {response.status_code})")
            return None

        data = response.json()
        result = data['chart']['result'][0]
        
        prices = result['indicators']['quote'][0]['close']
        df = pd.DataFrame({'Close': prices})
        df = df.dropna()
        
        if len(df) < 30:
            print(f"[{symbol}] データ件数が不足しています")
            return None

        # テクニカル指標の計算 (SMA20, RSI)
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # AI（機械学習）用の特徴量作成
        df['Return'] = df['Close'].pct_change()
        df['SMA_Diff'] = (df['Close'] - df['SMA20']) / df['SMA20']
        
        # 正解ラベル（翌日の終値が上がっていれば1、下がっていれば0）
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        df = df.dropna()
        
        if len(df) < 15:
            return None

        # 特徴量（RSI、騰落率、SMA乖離率）とターゲットの分割
        features = ['RSI', 'Return', 'SMA_Diff']
        X = df[features].iloc[:-1] # 最新の足以外を学習データにする
        y = df['Target'].iloc[:-1]

        # AIモデル（ロジスティック回帰）の学習
        model = LogisticRegression()
        model.fit(X, y)

        # 最新のデータで「明日上がる確率」を予測
        latest_X = df[features].iloc[[-1]]
        prob_up = model.predict_proba(latest_X)[0][1] * 100 # 上昇確率（%）

        current_price = df['Close'].iloc[-1]
        sma20 = df['SMA20'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        
        return {
            "symbol": symbol,
            "price": round(current_price, 2),
            "sma20": round(sma20, 2),
            "rsi": round(rsi, 1),
            "ai_prob": round(prob_up, 1)
        }
        
    except Exception as e:
        print(f"[{symbol}] エラー発生: {e}")
        return None

def analyze_and_notify():
    """データを解析し、AI予測とデモトレード判定を行ってDiscordに通知する"""
    embeds = []
    
    for symbol in SYMBOLS:
        data = get_crypto_data(symbol)
        if not data:
            continue
            
        price = data["price"]
        sma20 = data["sma20"]
        rsi = data["rsi"]
        ai_prob = data["ai_prob"]
        
        # AI予測確率とRSIを組み合わせた高度な判定ロジック
        if ai_prob >= 60.0 and rsi <= 40:
            signal = f"🤖🚀 【AI強気買いシグナル】"
            trade_action = f"AI上昇予測 {ai_prob}% ＆ 売られすぎ圏のため、仮想買い注文！"
            color = 3066993  # 緑色
        elif ai_prob <= 40.0 or rsi >= 70:
            signal = f"🤖⚠️ 【AI売り警戒・様子見】"
            trade_action = f"AI上昇予測低下 ({ai_prob}%) のため、ポジション調整・様子見"
            color = 15158332 # 赤色
        else:
            signal = f"🤖⚖️ 【AI中立・様子見】"
            trade_action = f"AI上昇予測 {ai_prob}%（明確なシグナルなし）"
            color = 3447003  # 青色

        # Discordの埋め込みメッセージ構築
        embed = {
            "title": f"🧠 AI予測相場判定: {symbol}",
            "color": color,
            "fields": [
                {"name": "AI判定シグナル", "value": signal, "inline": False},
                {"name": "自動売買アクション", "value": trade_action, "inline": False},
                {"name": "現在価格", "value": f"${price:,.2f}", "inline": True},
                {"name": "AI上昇予測", "value": f"{ai_prob}%", "inline": True},
                {"name": "RSI (14)", "value": f"{rsi}", "inline": True}
            ],
            "footer": {"text": f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }
        embeds.append(embed)

    if not embeds:
        print("通知するデータがありませんでした。")
        return

    # Discordへ送信（最大10件まで）
    payload = {"embeds": embeds[:10]}
    response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
    
    if response.status_code == 204:
        print("Discord通知が成功しました！")
    else:
        print(f"Discord通知に失敗しました: {response.status_code}, {response.text}")

if __name__ == "__main__":
    analyze_and_notify()
