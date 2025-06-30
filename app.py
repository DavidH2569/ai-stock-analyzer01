import streamlit as st
import pandas as pd
import requests
import openai
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import yfinance as yf

# Load API keys from Streamlit secrets
alpaca_key = st.secrets["ALPACA_KEY"]
alpaca_secret = st.secrets["ALPACA_SECRET"]
openai.api_key = st.secrets["OPENAI_KEY"]

# ----- FUNCTIONS -----
def fetch_top_100_tickers():
    url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
    params = {'scrIds': 'most_actives', 'count': '100', 'start': '0'}
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(url, headers=headers, params=params)
    data = res.json()
    quotes = data['finance']['result'][0]['quotes']
    return [q['symbol'] for q in quotes if 'symbol' in q]

def get_yahoo_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        df = df[['Close']]
        df['Return'] = df['Close'].pct_change()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df.dropna(inplace=True)
        return df
    except:
        return None


def predict_price(df):
    try:
        df['Target'] = df['Close'].shift(-10)
        df.dropna(inplace=True)
        X = df[['Close', 'Return', 'MA10', 'MA50']]
        y = df['Target']
        model = RandomForestRegressor()
        model.fit(X[:-10], y[:-10])
        pred = model.predict([X.iloc[-1]])[0]
        pct_gain = (pred - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100
        return round(pred, 2), round(pct_gain, 2)
    except:
        return None, None

def fetch_news(ticker):
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}"
    try:
        res = requests.get(url)
        news = res.json().get('news', [])
        if news:
            return news[0].get('title', '') + "\n" + news[0].get('summary', '')
    except:
        return "No news available."
    return "No news found."

def get_sentiment(news):
    prompt = f"Analyze the sentiment of this financial news. Is it positive, neutral, or negative?\n\n{news}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return "Sentiment analysis failed."

# ----- STREAMLIT UI -----
st.title("📊 AI Stock Analyzer")
st.write("Analyzing top 100 most active stocks using price prediction + news sentiment")

if st.button("Run Analysis"):
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "JPM", "V", "DIS"]
    results = []
    progress = st.progress(0)
    for i, ticker in enumerate(tickers):
        df = get_yahoo_data(ticker)
        if df is None or df.empty:
            continue
        pred_price, gain = predict_price(df)
        if pred_price is None:
            continue

        news = fetch_news(ticker)
        sentiment = get_sentiment(news)
        results.append({
            'Ticker': ticker,
            'Current Price': round(df['Close'].iloc[-1], 2),
            'Predicted Price': pred_price,
            '% Gain (10d)': gain,
            'News': news[:150],
            'Sentiment': sentiment[:150]
        })
        progress.progress((i+1)/len(tickers))
        
if results:
    df_result = pd.DataFrame(results)

    # Only sort if the column exists and contains numeric values
    if '% Gain (10d)' in df_result.columns:
        df_result = df_result[pd.to_numeric(df_result['% Gain (10d)'], errors='coerce').notnull()]
        df_result = df_result.sort_values(by='% Gain (10d)', ascending=False)

    st.subheader(f"✅ {len(df_result)} tickers analyzed successfully")
    st.dataframe(df_result, use_container_width=True)

    st.download_button("📤 Export CSV", df_result.to_csv(index=False), file_name="ai_stock_predictions.csv")
else:
    st.warning("⚠️ No data could be analyzed. Check your internet or API limits.")
