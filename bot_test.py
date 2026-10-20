import os
import requests
import pandas as pd
from datetime import datetime

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
    """Yahoo Financeからデータを取得する（鉄壁エラーガード付き）"""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=30d"
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
        
        if len(df) < 20:
            print(f"[{symbol}] データ件数が不足しています")
            return None

        # テクニカル指標の計算 (SMA20, RSI)
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        current_price = df['Close'].iloc[-1]
        sma20 = df['SMA20'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        
        return {
            "symbol": symbol,
            "price": round(current_price, 2),
            "sma20": round(sma20, 2),
            "rsi": round(rsi, 1)
        }
        
    except Exception as e:
        print(f"[{symbol}] エラー発生: {e}")
        return None

def analyze_and_notify():
    """データを解析し、デモトレードの売買判定を行ってDiscordに通知する"""
    embeds = []
    
    for symbol in SYMBOLS:
        data = get_crypto_data(symbol)
        if not data:
            continue
            
        price = data["price"]
        sma20 = data["sma20"]
        rsi = data["rsi"]
        
        # デモトレード（仮想売買）の判定ロジック
        if rsi <= 30:
            signal = "📉 【買いシグナル】"
            trade_action = "🤖 【デモ発動】売られすぎ圏のため、仮想買い注文を実行！"
            color = 3066993  # 緑色
        elif rsi >= 70:
            signal = "📈 【売り警戒】"
            trade_action = "🤖 【デモ発動】買われすぎ圏のため、仮想利食い売りを実行！"
            color = 15158332 # 赤色
        else:
            signal = "⚖️ 【中立・様子見】"
            trade_action = "💤 ポジション維持（売買なし）"
            color = 3447003  # 青色

        # Discordの埋め込みメッセージ（Embed）構築
        embed = {
            "title": f"🤖 デモトレード判定: {symbol}",
            "color": color,
            "fields": [
                {"name": "売買シグナル", "value": signal, "inline": False},
                {"name": "自動売買アクション", "value": trade_action, "inline": False},
                {"name": "現在価格", "value": f"${price:,.2f}", "inline": True},
                {"name": "SMA20", "value": f"${sma20:,.2f}", "inline": True},
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
