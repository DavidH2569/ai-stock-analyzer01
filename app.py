import streamlit as st
import pandas as pd
import yfinance as yf
import openai
from datetime import datetime, timedelta

# --- SETTINGS ---
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer (S&P 500)")

openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

# --- Load S&P 500 tickers ---
@st.cache_data
def load_sp500():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return table[0]["Symbol"].tolist()

def get_top_active_sp500(limit=50):
    tickers = load_sp500()
    volumes = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, period="5d", interval="1d", progress=False)
            if not data.empty:
                avg_volume = data["Volume"].mean()
                volumes.append((ticker, avg_volume))
        except:
            continue
    sorted_vols = sorted(volumes, key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_vols[:limit]]

# --- Data Fetching ---
def get_yahoo_data(ticker):
    try:
        return yf.download(ticker, period="30d", interval="1d", progress=False)
    except:
        return None

# --- Price Prediction ---
def predict_price(df):
    try:
        df = df.dropna()
        if len(df) < 10:
            return None, None
        recent = df["Close"][-10:].values
        pred_price = recent.mean() * 1.02  # Dummy model: +2% in 10d
        gain = ((pred_price - recent[-1]) / recent[-1]) * 100
        return pred_price, gain
    except:
        return None, None

# --- News Fetching ---
def fetch_news(ticker):
    try:
        return f"Latest news for {ticker}."  # Replace with real news fetch logic later
    except:
        return "No news available."

# --- Sentiment Analysis ---
def get_sentiment(news):
    if not news or "no news" in news.lower():
        return "No news"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Analyze the sentiment of this financial news: {news}"}]
        )
        return response["choices"][0]["message"]["content"]
    except:
        return "Sentiment failed"

# --- MAIN ---
st.subheader("Step 1: Get Top 50 Active S&P 500 Tickers")
if st.button("Run Analysis"):
    with st.spinner("📥 Loading tickers and analyzing..."):
        tickers = get_top_active_sp500(limit=50)
        results = []

        for ticker in tickers:
            df = get_yahoo_data(ticker)
            if df is None or df.empty:
                continue

            pred_price, gain = predict_price(df)
            if pred_price is None:
                continue

            try:
                current_price = df['Close'].iloc[-1]
                gain_float = float(gain)
            except:
                continue

            news = fetch_news(ticker)
            sentiment = get_sentiment(news)

            results.append({
                "Ticker": ticker,
                "Current Price": round(current_price, 2),
                "Predicted Price": round(pred_price, 2),
                "% Gain (10d)": round(gain_float, 2),
                "News": news,
                "Sentiment": sentiment
            })

        df_result = pd.DataFrame(results)

        if not df_result.empty:
            df_result = df_result[pd.to_numeric(df_result['% Gain (10d)'], errors='coerce').notnull()]
            df_result = df_result[df_result['% Gain (10d)'] > 0]
            df_result = df_result.sort_values(by='% Gain (10d)', ascending=False)

            st.success(f"✅ {len(df_result)} stocks found with positive forecast")
            st.dataframe(df_result, use_container_width=True)
            st.download_button("📤 Download Results", df_result.to_csv(index=False), "ai_predictions.csv")
        else:
            st.warning("⚠️ No stocks found with positive predicted gains.")


