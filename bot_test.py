import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
import ccxt
import traceback

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

def send_discord_error(error_msg):
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        payload = {
            "content": "🚨 **【要塞システム・緊急エラー警報】** エラーが発生しました！確認してください。 @here",
            "embeds": [{
                "title": "⚠️ クラッシュ・例外エラー発生",
                "color": 15158332,
                "fields": [{"name": "エラー詳細", "value": f"```{error_msg[:1000]}```", "inline": False}],
                "footer": {"text": f"発生日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
            }]
        }
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        print(f"Discordエラー通知送信失敗: {e}")

def fetch_bitflyer_balance_info():
    if not EXCHANGE_API_KEY or not EXCHANGE_API_SECRET:
        return "💡 API未設定 (シミュレーションモード稼働中)"

    try:
        exchange = ccxt.bitflyer({
            'apiKey': EXCHANGE_API_KEY,
            'secret': EXCHANGE_API_SECRET,
            'enableRateLimit': True,
        })
        balance = exchange.fetch_balance()
        jpy_cash = balance['total'].get('JPY', 0.0)
        
        total_asset_jpy = jpy_cash
        assets_detail = []
        
        for curr, amount in balance['total'].items():
            if curr != 'JPY' and amount > 0:
                try:
                    ticker = exchange.fetch_ticker(f"{curr}/JPY")
                    price = ticker['last']
                    val = amount * price
                    total_asset_jpy += val
                    assets_detail.append(f"{curr}: {amount:.4f} (約¥{val:,.0f})")
                except Exception:
                    pass
                    
        detail_str = f" [現金: ¥{jpy_cash:,.0f}]" if assets_detail else ""
        return f"¥{total_asset_jpy:,.0f}{detail_str}"
    except Exception as e:
        return f"残高取得エラー: {str(e)[:30]}"

def fetch_chart_data(yf_symbol, timeframe, limit=120):
    try:
        if timeframe == '1d':
            interval, range_val = '1d', '180d'
        elif timeframe == '1h':
            interval, range_val = '1h', '14d'
        elif timeframe == '15m':
            interval, range_val = '15m', '5d'
        else:
            interval, range_val = '1h', '7d'

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_symbol}?interval={interval}&range={range_val}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        data = res.json()['chart']['result'][0]
        
        prices = data['indicators']['quote'][0]['close']
        volumes = data['indicators']['quote'][0]['volume']
        
        df = pd.DataFrame({'Close': prices, 'Volume': volumes}).dropna()
        return df
    except Exception as e:
        print(f"[{yf_symbol}] データ取得エラー({timeframe}): {e}")
        return None

def fetch_macro_data():
    try:
        usdjpy = fetch_chart_data("USDJPY=X", "1d", limit=30)
        sp500 = fetch_chart_data("^GSPC", "1d", limit=30)
        
        macro_score = 0
        if usdjpy is not None and len(usdjpy) > 5:
            if usdjpy['Close'].iloc[-1] > usdjpy['Close'].rolling(5).mean().iloc[-1]:
                macro_score += 1
        if sp500 is not None and len(sp500) > 5:
            if sp500['Close'].iloc[-1] > sp500['Close'].rolling(5).mean().iloc[-1]:
                macro_score += 1
                
        return macro_score
    except Exception:
        return 1

def execute_bitflyer_order(symbol, amount_jpy=1000):
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

        return f"API接続OK / 発注スタンバイ (数量: {amount})"
    except Exception as e:
        return f"国内API確認中: {str(e)[:35]}"

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

def run_backtest_with_trailing(df_daily, all_probs):
    """厳格な損切り・トレールに加え、最大ドローダウンを算出するバックテスト"""
    trades = []
    fee = 0.002
    equity = 1.0
    equity_curve = [1.0]
    
    for i in range(len(all_probs) - 1):
        prob = all_probs[i]
        rsi = df_daily['RSI'].iloc[i]
        
        if prob >= 50.0 and rsi <= 60:
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
            equity *= (1 + net_ret)
            equity_curve.append(equity)
            
    if not trades:
        return {"total_trades": 0, "win_rate": 0.0, "total_return": 0.0, "max_drawdown": 0.0}
        
    wins = [t for t in trades if t > 0]
    win_rate = (len(wins) / len(trades)) * 100
    total_return = (np.prod([1 + t for t in trades]) - 1) * 100
    
    # 最大ドローダウン（最高値からの最大下落率）の算出
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak
        if dd < max_dd:
            max_dd = dd
    max_drawdown = round(max_dd * 100, 1)
    
    return {
        "total_trades": len(trades),
        "win_rate": round(win_rate, 1),
        "total_return": round(total_return, 1),
        "max_drawdown": max_drawdown
    }

def save_log_to_csv(data_list):
    """判定結果をCSVファイルに蓄積保存する関数"""
    csv_file = "bot_history_log.csv"
    rows = []
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    for d in data_list:
        rows.append({
            "Datetime": now_str,
            "Symbol": d["symbol"],
            "Price": d["price"],
            "AI_Prob": d["ai_prob"],
            "WinRate": d["bt_win_rate"],
            "TotalReturn": d["bt_return"],
            "MaxDD": d["max_drawdown"],
            "Status": d["order_status"]
        })
    df_new = pd.DataFrame(rows)
    try:
        if os.path.exists(csv_file):
            df_new.to_csv(csv_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df_new.to_csv(csv_file, mode='w', header=True, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"CSV保存エラー: {e}")

def analyze_crypto(item, macro_score):
    symbol = item["symbol"]
    yf_symbol = item["yf"]
    
    df_daily = fetch_chart_data(yf_symbol, "1d", limit=150)
    df_1h = fetch_chart_data(yf_symbol, "1h", limit=168)
    df_15m = fetch_chart_data(yf_symbol, "15m", limit=96)
    
    if df_daily is None or len(df_daily) < 40:
        return None

    df_daily = process_features(df_daily)
    df_daily['Macro_Score'] = macro_score
    df_daily['Target'] = (df_daily['Close'].shift(-1) > df_daily['Close']).astype(int)
    df_clean = df_daily.dropna()
    
    if len(df_clean) < 25:
        return None

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
        hourly_trend = "上昇" if latest_1h_close > latest_1h_sma else "下降"

    m15_trend = "不明"
    if df_15m is not None and len(df_15m) >= 20:
        df_15m['SMA20'] = df_15m['Close'].rolling(window=20).mean()
        latest_15m_close = df_15m['Close'].iloc[-1]
        latest_15m_sma = df_15m['SMA20'].iloc[-1]
        m15_trend = "短期上昇" if latest_15m_close > latest_15m_sma else "短期押し目"

    all_probs = model.predict_proba(df_clean[features])[:, 1] * 100
    bt_result = run_backtest_with_trailing(df_clean, all_probs)

    current_price = df_clean['Close'].iloc[-1]
    rsi = df_clean['RSI'].iloc[-1]
    bb_upper = df_clean['BB_Upper'].iloc[-1]
    
    stop_loss = current_price * 0.97
    trailing_target = current_price * 1.025

    is_strong_signal = (
        prob_daily >= 53.0 and 
        ("上昇" in hourly_trend or "上昇" in m15_trend) and 
        macro_score >= 1 and 
        rsi <= 55 and 
        current_price <= bb_upper
    )

    order_status = "見送り"
    if is_strong_signal:
        order_status = execute_bitflyer_order(symbol, amount_jpy=1000)
        send_line_notify(f"\n🔥【アグレッシブシグナル点灯】\n銘柄: {symbol}\n価格: ¥{current_price:,.0f}\nAI確率: {prob_daily}%\n15m状態: {m15_trend}")

    return {
        "symbol": symbol,
        "price": round(current_price, 2),
        "rsi": round(rsi, 1),
        "ai_prob": round(prob_daily, 1),
        "hourly_trend": hourly_trend,
        "m15_trend": m15_trend,
        "macro_score": macro_score,
        "stop_loss": round(stop_loss, 2),
        "trailing_target": round(trailing_target, 2),
        "bt_trades": bt_result["total_trades"],
        "bt_win_rate": bt_result["win_rate"],
        "bt_return": bt_result["total_return"],
        "max_drawdown": bt_result["max_drawdown"],
        "order_status": order_status,
        "is_signal": is_strong_signal
    }

def analyze_and_notify():
    macro_score = fetch_macro_data()
    balance_info = fetch_bitflyer_balance_info()
    embeds = []
    all_results = []
    has_strong_signal = False
    
    for item in SYMBOLS:
        data = analyze_crypto(item, macro_score)
        if not data:
            continue
        all_results.append(data)
            
        symbol = data["symbol"]
        price = data["price"]
        rsi = data["rsi"]
        ai_prob = data["ai_prob"]
        hourly_trend = data["hourly_trend"]
        m15_trend = data["m15_trend"]
        stop_loss = data["stop_loss"]
        trailing_target = data["trailing_target"]
        order_status = data["order_status"]
        
        if data["is_signal"]:
            has_strong_signal = True
            signal = "🔥🚀 【アグレッシブ要塞・攻め買いシグナル】"
            trade_action = f"⚡ {order_status}"
            color = 3066993
        elif ai_prob <= 42.0 or macro_score == 0:
            signal = "🛡️⚠️ 【要塞防御・マクロ警戒見送り】"
            trade_action = "世界経済またはAI確率低下により見送り"
            color = 15158332
        else:
            signal = "⚡⚖️ 【中立・タイミング待ち】"
            trade_action = "明確なエントリー条件未達"
            color = 3447003

        embed = {
            "title": f"🇯🇵 アグレッシブ連動アナリティクス: {symbol}",
            "color": color,
            "fields": [
                {"name": "💰 口座総資産(bitFlyer)", "value": f"**{balance_info}**", "inline": False},
                {"name": "判定シグナル", "value": signal, "inline": False},
                {"name": "自動売買アクション", "value": trade_action, "inline": False},
                {"name": "現在価格", "value": f"¥{price:,.2f}", "inline": True},
                {"name": "日足AI予測(マイルド)", "value": f"{ai_prob}%", "inline": True},
                {"name": "マルチタイム分析", "value": f"1H:{hourly_trend} / 15M:{m15_trend}", "inline": True},
                {"name": "📈 過去検証(勝率/回数)", "value": f"{data['bt_win_rate']}% ({data['bt_trades']}回)", "inline": True},
                {"name": "💰 過去通算リターン", "value": f"{data['bt_return']}%", "inline": True},
                {"name": "📉 最大下落耐性(MaxDD)", "value": f"{data['max_drawdown']}%", "inline": True},
                {"name": "RSI (14)", "value": f"{rsi}", "inline": True},
                {"name": "🛑 防御損切り (-3%)", "value": f"¥{stop_loss:,.2f}", "inline": True},
                {"name": "🔄 トレール起動地点", "value": f"¥{trailing_target:,.2f}", "inline": True}
            ],
            "footer": {"text": f"実行日時(完全要塞プロ版V3): {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }
        embeds.append(embed)

    # CSVへのログ保存を実行
    if all_results:
        save_log_to_csv(all_results)

    if not embeds:
        print("有効なデータがありませんでした")
        return

    content_msg = "🔥 **【アグレッシブアラート】** 攻めのエントリーシグナルが点灯しました！ @here" if has_strong_signal else None

    payload = {"embeds": embeds[:10]}
    if content_msg:
        payload["content"] = content_msg

    response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
    if response.status_code == 204:
        print("完全要塞プロ版V3 通知成功＆CSV記録完了！")
    else:
        print(f"通知失敗: {response.status_code}")

if __name__ == "__main__":
    try:
        analyze_and_notify()
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"致命的エラー発生: {error_trace}")
        send_discord_error(error_trace)
        send_line_notify(f"🚨【要塞システム緊急停止】エラーが発生しました。詳細はDiscordを確認してください。")
        raise e
