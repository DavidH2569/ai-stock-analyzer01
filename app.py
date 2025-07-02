import streamlit as st
import pandas as pd
import yfinance as yf
import openai
import os

# Set your GPT key here or use Streamlit secrets
openai.api_key = st.secrets.get("OPENAI_KEY", "")

st.title("📈 AI Stock Analyzer")

# Example hardcoded tickers (20 popular ones)
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "V", "DIS",
    "NFLX", "PEP", "KO", "INTC", "AMD",
    "BA", "WMT", "CVX", "XOM", "PFE"
]

def get_yahoo_data(ticker):
    try:
        df = yf.download(ticker, period="60d", interval="1d", progress=False)
        return df if not df.empty else None
    except Exception:
        return None

def predict_price(df):
    try:
        if len(df) < 10:
            return None, None
        recent_close = df["Close"].iloc[-1]
        avg_past_10 = df["Close"].iloc[-11:-1].mean()
        gain = ((avg_past_10 - recent_close) / recent_close) * 100
        return avg_past_10, gain
    except:
        return None, None

def get_sentiment(news):
    if not news or news.strip() == "":
        return "No news"
    try:
        prompt = f"Analyze the sentiment of this financial news headline:\n\n{news}\n\nRespond with Positive, Neutral, or Negative."
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Sentiment failed"

if st.button("▶️ Run Analysis"):
    st.info("Running analysis on 20 S&P 500 stocks...")

    results = []
    for ticker in TICKERS:
        # st.write(f"🔍 Analyzing {ticker}...")
        df = get_yahoo_data(ticker)
        if df is None:
            continue
        pred_price, gain = predict_price(df)
        if gain is None:
            continue

        news = f"Recent news about {ticker} stock."  # Placeholder
        sentiment = get_sentiment(news)

        results.append({
            "Ticker": ticker,
            'Current Price': round(float(df['Close'].iloc[-1]), 2),
            "Predicted Price": round(float(pred_price), 2) if pred_price is not None else None,
            "% Gain (10d)": round(float(gain), 2) if gain is not None else None,
            "Sentiment": sentiment
        })

    if results:
        df_result = pd.DataFrame(results)
        df_result = df_result[df_result["% Gain (10d)"] > 0]
        df_result = df_result.sort_values(by="% Gain (10d)", ascending=False)

        if not df_result.empty:
            st.subheader(f"📊 {len(df_result)} Positive Tickers")
            st.dataframe(df_result, use_container_width=True)
            st.download_button("💾 Export CSV", df_result.to_csv(index=False), file_name="stock_results.csv")
        else:
            st.warning("⚠️ No tickers with positive predicted gains.")
    else:
        st.error("❌ No data was analyzed.")


