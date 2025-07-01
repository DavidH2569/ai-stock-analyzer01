import streamlit as st
import pandas as pd
import yfinance as yf
import openai
import time

st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI Stock Analyzer - S&P 500")

openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

@st.cache_data(show_spinner=False)
def load_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)
    return table[0]["Symbol"].tolist()

@st.cache_data(show_spinner=False)
def get_top_active_sp500(limit=20):
    tickers = load_sp500()
    volumes = []
    for i, ticker in enumerate(tickers):
        try:
            data = yf.download(ticker, period="5d", progress=False)
            vol = data["Volume"].iloc[-1] if not data.empty else 0
            volumes.append((ticker, vol))
        except:
            volumes.append((ticker, 0))
        time.sleep(0.5)
    sorted_volumes = sorted(volumes, key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_volumes[:limit]]

def get_yahoo_data(ticker):
    try:
        return yf.download(ticker, period="60d", interval="1d", progress=False)
    except:
        return None

def predict_price(df):
    try:
        last_close = df["Close"].iloc[-1]
        future_close = df["Close"].iloc[-1] * (1 + (df["Close"].pct_change().mean() * 10))
        pct_gain = ((future_close - last_close) / last_close) * 100
        return future_close, pct_gain
    except:
        return None, None

def get_sentiment(news):
    if not news or news.strip() == "" or news == "No news available.":
        return "No news to analyze."
    prompt = f"Analyze the sentiment of this financial news. Is it positive, neutral, or negative?\n\n{news}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return "Sentiment analysis failed."

def fetch_news(ticker):
    return "Placeholder news text for sentiment analysis."

st.sidebar.header("⚙️ Options")
if st.sidebar.button("Run Analysis"):
    st.subheader("📥 Loading S&P 500 tickers...")
    tickers = get_top_active_sp500(limit=20)
    results = []

    for i, ticker in enumerate(tickers):
        st.write(f"🔄 Checking {ticker} ({i+1}/{len(tickers)})")

        df = get_yahoo_data(ticker)
        if df is None or df.empty:
            continue

        pred_price, gain = predict_price(df)
        if pred_price is None:
            continue

        news = fetch_news(ticker)
        sentiment = get_sentiment(news)

        try:
            gain_float = float(gain)
        except (TypeError, ValueError):
            gain_float = None

        results.append({
            'Ticker': ticker,
            'Current Price': round(df['Close'].iloc[-1], 2),
            'Predicted Price': round(pred_price, 2) if pred_price is not None else None,
            '% Gain (10d)': round(gain_float, 2) if gain_float is not None else None,
            'News': news[:150],
            'Sentiment': sentiment[:150]
        })

    if results:
        df_result = pd.DataFrame(results)
        if '% Gain (10d)' in df_result.columns:
            df_result['% Gain (10d)'] = pd.to_numeric(df_result['% Gain (10d)'], errors='coerce')
            df_result = df_result.dropna(subset=['% Gain (10d)'])
            df_result = df_result[df_result['% Gain (10d)'] > 0]
            df_result = df_result.sort_values(by='% Gain (10d)', ascending=False)

        if df_result.empty:
            st.warning("⚠️ No tickers with positive predicted gains today.")
        else:
            st.subheader(f"✅ {len(df_result)} tickers analyzed successfully")
            st.dataframe(df_result, use_container_width=True)
            st.download_button("📤 Export CSV", df_result.to_csv(index=False), file_name="ai_stock_predictions.csv")
    else:
        st.warning("⚠️ No data could be analyzed. Check yfinance response or prediction logic.")
else:
    st.info("👈 Click 'Run Analysis' to begin.")


