from flask import Flask, render_template, request, redirect, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
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
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your_secret_key")

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    stock_symbol = db.Column(db.String(10), nullable=False)
    predicted_graph = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        action = request.form["action"]
        if action == "signup":
            username = request.form["username"]
            email = request.form["email"]
            password = generate_password_hash(request.form["password"])
            if User.query.filter_by(email=email).first():
                flash("Email already registered! Try logging in.", "danger")
            else:
                new_user = User(username=username, email=email, password=password)
                db.session.add(new_user)
                db.session.commit()
                flash("Signup successful! Please login.", "success")
        elif action == "login":
            email = request.form["email"]
            password = request.form["password"]
            user = User.query.filter_by(email=email).first()
            if user and check_password_hash(user.password, password):
                session["user_id"] = user.id
                return redirect("/stock")
            else:
                flash("Invalid credentials! Try again.", "danger")
    return render_template("login.html")

def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="2y")
    info = stock.info
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
    if "user_id" not in session:
        return redirect("/")
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

        db.session.add(PredictionHistory(user_id=session["user_id"], stock_symbol=symbol1, predicted_graph=plot1))
        if symbol2:
            db.session.add(PredictionHistory(user_id=session["user_id"], stock_symbol=symbol2, predicted_graph=plot2))
        db.session.commit()

        return render_template("index.html", symbol1=symbol1, current_price1=current_price1, predicted_price1=predicted_price1, plot1=plot1, info1=info1, news1=news1, symbol2=symbol2, current_price2=current_price2, predicted_price2=predicted_price2, plot2=plot2, info2=info2, news2=news2)

    return render_template("index.html", symbol1=None, symbol2=None)

@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect("/")
    user_history = PredictionHistory.query.filter_by(user_id=session["user_id"]).order_by(PredictionHistory.timestamp.desc()).all()
    return render_template("history.html", history=user_history)

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
