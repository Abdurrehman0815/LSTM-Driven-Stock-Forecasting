from flask import Flask, render_template, request, redirect, session, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import feedparser
from datetime import datetime
import os
import secrets
import threading
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your_secret_key")

# Prediction history is kept in server memory, one list per browser session.
# The browser only holds a random session ID in its cookie.
# History is lost when the server restarts or redeploys.
HISTORY_LOCK = threading.Lock()
HISTORY_STORE = {}  # session id -> list of entries (oldest first)
MAX_ENTRIES_PER_SESSION = 20
MAX_SESSIONS = 500

def get_session_id():
    if "sid" not in session:
        session["sid"] = secrets.token_hex(16)
    return session["sid"]

def save_history(symbol, plot_html):
    sid = get_session_id()
    entry = {
        "stock_symbol": symbol,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_graph": plot_html,
    }
    with HISTORY_LOCK:
        entries = HISTORY_STORE.setdefault(sid, [])
        entries.append(entry)
        del entries[:-MAX_ENTRIES_PER_SESSION]
        while len(HISTORY_STORE) > MAX_SESSIONS:
            HISTORY_STORE.pop(next(iter(HISTORY_STORE)))

def get_history():
    sid = get_session_id()
    with HISTORY_LOCK:
        return list(reversed(HISTORY_STORE.get(sid, [])))

@app.route("/")
def home():
    return redirect("/stock")

def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="2y")
    # Yahoo's "info" call often returns nothing from cloud hosts such as Render,
    # so fill the price fields from the price history, which still loads.
    try:
        info = dict(stock.info or {})
    except Exception:
        info = {}
    if not data.empty:
        closes = data['Close']
        if not info.get('currentPrice'):
            info['currentPrice'] = round(float(closes.iloc[-1]), 2)
        if not info.get('previousClose') and len(closes) > 1:
            info['previousClose'] = round(float(closes.iloc[-2]), 2)
        if not info.get('open'):
            info['open'] = round(float(data['Open'].iloc[-1]), 2)
        last_year = data.tail(252)
        if not info.get('fiftyTwoWeekHigh'):
            info['fiftyTwoWeekHigh'] = round(float(last_year['High'].max()), 2)
        if not info.get('fiftyTwoWeekLow'):
            info['fiftyTwoWeekLow'] = round(float(last_year['Low'].min()), 2)
    return data, info

def predict_stock(symbol):
    data, info = get_stock_data(symbol)
    if data.empty:
        return None, None, None, None

    df = data[['Close']].copy()
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['Close', 'Target']])
    X = df_scaled[:-1, 0].reshape(-1, 1)
    y = df_scaled[:-1, 1]

    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X, y)

    last_close = df_scaled[-1, 0]
    future_prices = []
    for _ in range(60):
        pred_scaled = model.predict(np.array([[last_close]]))[0]
        future_prices.append(pred_scaled)
        last_close = pred_scaled

    future_prices = scaler.inverse_transform(np.array([[0, p] for p in future_prices]))[:, 1]
    current_price = round(info.get('currentPrice', 0), 2)

    return data, info, future_prices, current_price

def plot_stock(data, future_prices, color):
    past_trace = go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Past Prices', line=dict(color=color))
    future_dates = pd.date_range(start=data.index[-1], periods=60, freq='D')
    future_trace = go.Scatter(x=future_dates, y=future_prices, mode='lines', name='Predicted Prices', line=dict(color='red', dash='dot'))
    layout = go.Layout(xaxis=dict(title="Date"), yaxis=dict(title="Price (USD)"), plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))
    fig = go.Figure(data=[past_trace, future_trace], layout=layout)
    return pyo.plot(fig, output_type='div')

def fetch_news(symbol):
    url = f"https://news.google.com/rss/search?q={symbol}+stock+market"
    news_feed = feedparser.parse(url)
    return [{"title": entry.title, "link": entry.link} for entry in news_feed.entries[:5]]

@app.route('/news')
def news():
    symbol1 = request.args.get('symbol1')
    symbol2 = request.args.get('symbol2')
    news_list = fetch_news(symbol1) if symbol1 else []
    if symbol2:
        news_list += fetch_news(symbol2)
    return render_template('news.html', symbol=symbol1 or symbol2, news_list=news_list)

@app.route("/stock", methods=["GET", "POST"])
def stock():
    symbol1 = symbol2 = None
    if request.method == "POST":
        symbol1 = request.form["symbol1"].upper()
        symbol2 = request.form.get("symbol2", "").upper()

        data1, info1, future_prices1, current_price1 = predict_stock(symbol1)
        if data1 is None:
            return render_template("index.html", error="Invalid stock symbol!")

        plot1 = plot_stock(data1, future_prices1, "cyan")
        predicted_price1 = round(future_prices1[-1], 2)
        news1 = fetch_news(symbol1)

        plot2 = predicted_price2 = current_price2 = info2 = news2 = None
        if symbol2:
            data2, info2, future_prices2, current_price2 = predict_stock(symbol2)
            if data2 is not None:
                plot2 = plot_stock(data2, future_prices2, "orange")
                predicted_price2 = round(future_prices2[-1], 2)
                news2 = fetch_news(symbol2)

        save_history(symbol1, plot1)
        if plot2 is not None:
            save_history(symbol2, plot2)

        return render_template("index.html", symbol1=symbol1, current_price1=current_price1, predicted_price1=predicted_price1, plot1=plot1, info1=info1, news1=news1, symbol2=symbol2, current_price2=current_price2, predicted_price2=predicted_price2, plot2=plot2, info2=info2, news2=news2)

    return render_template("index.html", symbol1=None, symbol2=None)

@app.route("/history")
def history():
    return render_template("history.html", history=get_history())

@app.route('/get_stock_prices')
def get_stock_prices():
    symbols = ["COALINDIA.NS", "SUNPHARMA.NS", "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", "AXISBANK.NS", "NESTLEIND.NS", "LT.NS", "KOTAKBANK.NS", "TATAMOTORS.NS", "EICHERMOT.NS", "POWERGRID.NS", "DRREDDY.NS", "ICICIBANK.NS", "SHRIRAMFIN.NS", "ADANIENT.NS", "HEROMOTOCO.NS", "ADANIPORTS.NS", "HDFCLIFE.NS", "BHARTIARTL.NS", "JSWSTEEL.NS", "INDUSINDBK.NS", "RELIANCE.NS", "SBIN.NS", "ITC.NS", "ASIANPAINT.NS"]
    stock_data = []
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        data = stock.history(period="2d")
        if not data.empty:
            current_price = round(stock.fast_info["last_price"], 2)
            previous_close = round(stock.fast_info["previous_close"], 2)
            change = round(current_price - previous_close, 2)
            percent = round((change / previous_close) * 100, 2)
            stock_data.append({"symbol": symbol.replace(".NS", ""), "current_price": current_price, "price_change": change, "percentage_change": percent})
    return jsonify(stock_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
