import streamlit as st
import pandas as pd
import requests
import openai
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import yfinance as yf

# Load API keys from Streamlit secrets
# alpaca_key = st.secrets["ALPACA_KEY"]
# alpaca_secret = st.secrets["ALPACA_SECRET"]
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
        if len(df) < 50:
            return None, None
        df['Target'] = df['Close'].shift(-10)
        df.dropna(inplace=True)
        X = df[['Close', 'Return', 'MA10', 'MA50']]
        y = df['Target']
        model = RandomForestRegressor()
        model.fit(X[:-10], y[:-10])
        pred = float(model.predict([X.iloc[-1]])[0])
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
    if not news or news.strip() == "" or news == "No news available.":
        return "No news to analyze."

    prompt = f"Analyze the sentiment of this financial news. Is it positive, neutral, or negative?\n\n{news}"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception:
        return "Sentiment analysis failed."

def get_top_active_sp500(limit=100):
    # import requests
    # import pandas as pd

    st.write("📥 Loading S&P 500 tickers...")
    sp500_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    df = pd.read_csv(sp500_url)
    tickers = df['Symbol'].tolist()

    volumes = []
    for i, ticker in enumerate(tickers):
        try:
            st.write(f"🔄 Checking {ticker} ({i+1}/{len(tickers)})")
            data = yf.download(ticker, period="5d", interval="1d", progress=False)
            if 'Volume' in data.columns and not data['Volume'].isnull().all():
                avg_volume = data['Volume'].mean()
                if pd.notna(avg_volume):
                    volumes.append((ticker, float(avg_volume)))
        except Exception:
            continue

    st.write("✅ Finished collecting volumes. Sorting...")
    sorted_volumes = sorted(volumes, key=lambda x: x[1], reverse=True)
    top_tickers = [t[0] for t in sorted_volumes[:limit]]
    st.write(f"✅ Top {limit} tickers selected.")
    return top_tickers


# ----- STREAMLIT UI -----
st.title("📊 AI Stock Analyzer")
st.write("Analyzing top 50 most active stocks using price prediction + news sentiment")

if st.button("Run Analysis"):
    tickers = get_top_active_sp500(limit=50)
    results = []

    progress = st.progress(0)

    for i, ticker in enumerate(tickers):
        st.write(f"🔍 Analyzing {ticker}...")

        df = get_yahoo_data(ticker)
        if df is None or df.empty:
            st.write(f"❌ {ticker}: No price data")
            continue

        pred_price, gain = predict_price(df)
        if pred_price is None:
            st.write(f"❌ {ticker}: Prediction failed")
            continue

        news = fetch_news(ticker)
        sentiment = get_sentiment(news)

        try:
            gain_float = float(gain)
            st.write(f"✅ {ticker}: Gain {gain_float:.2f}%")
        except:
            st.write(f"⚠️ {ticker}: Gain conversion failed")

        results.append({
            'Ticker': ticker,
            'Current Price': round(df['Close'].iloc[-1], 2),
            'Predicted Price': float(pred_price),
            '% Gain (10d)': gain_float,
            'News': news[:150],
            'Sentiment': sentiment[:150]
        })



        progress.progress((i + 1) / len(tickers))

    if results:
        df_result = pd.DataFrame(results)

        if '% Gain (10d)' in df_result.columns:
            df_result['% Gain (10d)'] = pd.to_numeric(df_result['% Gain (10d)'], errors='coerce')
    
            # 🔍 Debug: See all predicted gains before filtering
            # st.write("🔎 Raw predicted gains before filtering:", df_result[['Ticker', '% Gain (10d)']])
    
            df_result = df_result.dropna(subset=['% Gain (10d)'])
            df_result = df_result[df_result['% Gain (10d)'] > 0.01]
            df_result = df_result.sort_values(by='% Gain (10d)', ascending=False)


            if len(df_result) > 0:
                st.subheader(f"✅ {len(df_result)} tickers with positive predicted gain")
                st.dataframe(df_result, use_container_width=True)
                st.download_button("📤 Export CSV", df_result.to_csv(index=False), file_name="ai_stock_predictions.csv")
            else:
                st.warning("⚠️ No tickers with positive predicted gains today.")
        else:
            st.warning("⚠️ Prediction results missing expected column.")
    else:
        st.warning("⚠️ No data could be analyzed. Check yfinance response or prediction logic.")


