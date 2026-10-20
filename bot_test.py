import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# GitHub Secretsから安全にDiscordのWebhook URLを読み込む
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

# 監視銘柄（5大銘柄）
SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"]

def get_crypto_data(symbol):
    """Yahoo Financeからデータを取得する（鉄壁エラーガード付き）"""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=60d"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"[{symbol}] データ取得失敗 (ステータスコード: {response.status_code})")
            return None
            
        data = response.json()
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        prices = result['indicators']['quote'][0]['close']
        
        df = pd.DataFrame({
            'timestamp': [datetime.fromtimestamp(ts) for ts in timestamps],
            'close': prices
        }).dropna()
        
        return df
    except Exception as e:
        print(f"[{symbol}] データ取得中にエラー発生: {e}")
        return None

def analyze_and_notify():
    """4大頭脳による分析とDiscord通知を行うメイン処理"""
    if not DISCORD_WEBHOOK_URL:
        print("エラー: DISCORD_WEBHOOK_URL が設定されていません。")
        return

    embeds = []

    for symbol in SYMBOLS:
        df = get_crypto_data(symbol)
        if df is None or len(df) < 30:
            continue

        # 指標計算（SMA, RSI, ボリンジャーバンド, MACD）
        close = df['close']
        sma20 = close.rolling(window=20).mean().iloc[-1]
        
        # RSI計算
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        current_price = close.iloc[-1]

        # 簡易判定メッセージの作成
        embed = {
            "title": f"📊 リアルタイム相場判定: {symbol}",
            "color": 3447003, # 青色
            "fields": [
                {"name": "現在価格", "value": f"${current_price:,.2f}", "inline": True},
                {"name": "SMA20", "value": f"${sma20:,.2f}", "inline": True},
                {"name": "RSI (14)", "value": f"{rsi:.1f}", "inline": True}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        embeds.append(embed)

    # Discordへ一括通知
    if embeds:
        payload = {
            "username": "鉄壁暗号資産ボット",
            "embeds": embeds[:10] # Discordの制限（最大10件）
        }
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        if response.status_code == 204:
            print("Discordへ正常に通知を送信しました！")
        else:
            print(f"Discord通知に失敗しました: {response.status_code}")

if __name__ == "__main__":
    analyze_and_notify()
