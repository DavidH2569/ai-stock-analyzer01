import streamlit as st
import pandas as pd
import requests
import openai
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import numpy as np

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

def get_alpaca_data(ticker):
    url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars?timeframe=1Day&limit=100"
    headers = {
        'APCA-API-KEY-ID': alpaca_key,
        'APCA-API-SECRET-KEY': alpaca_secret
    }
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        return None
    raw = res.json()['bars']
    df = pd.DataFrame(raw)
    df['t'] = pd.to_datetime(df['t'])
    df.set_index('t', inplace=True)
    df.rename(columns={'c': 'Close'}, inplace=True)
    df['Return'] = df['Close'].pct_change()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    return df.dropna()

def predict_price(df):
    df['Target'] = df['Close'].shift(-10)
    df.dropna(inplace=True)
    X = df[['Close', 'Return', 'MA10', 'MA50']]
    y = df['Target']
    model = RandomForestRegressor()
    model.fit(X[:-10], y[:-10])
    pred = model.predict([X.iloc[-1]])[0]
    pct_gain = (pred - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100
    return round(pred, 2), round(pct_gain, 2)

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
    tickers = fetch_top_100_tickers()
    results = []
    progress = st.progress(0)
    for i, ticker in enumerate(tickers):
        df = get_alpaca_data(ticker)
        if df is None or df.empty:
            continue
        pred_price, gain = predict_price(df)
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

    df_result = pd.DataFrame(results).sort_values(by='% Gain (10d)', ascending=False)
    st.dataframe(df_result, use_container_width=True)
    st.download_button("📤 Export CSV", df_result.to_csv(index=False), file_name="ai_stock_predictions.csv")
