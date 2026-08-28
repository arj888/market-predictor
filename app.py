import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Price Predictor", layout="centered")
st.title("Market Predictor")

ticker = st.text_input("Enter Ticker (e.g. BTC-USD, RELIANCE.NS, NVDA):", "").strip().upper()

if st.button("Predict"):
    if not ticker:
        st.error("Please enter a valid ticker.")
    else:
        with st.spinner("Calculating..."):
            df = yf.download(ticker, period="2y", interval="1d", auto_adjust=True, progress=False)

            if df.empty:
                st.error("Invalid Ticker. Data not found.")
            else:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df['Daily_Return'] = df['Close'].pct_change()
                df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()
                df['SMA_10'] = df['Close'].rolling(window=10).mean()
                df['SMA_50'] = df['Close'].rolling(window=50).mean()

                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-9)
                df['RSI_14'] = 100 - (100 / (1 + rs))

                df['Target'] = df['Close'].shift(-1)

                features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'Volatility_20', 'SMA_10', 'SMA_50', 'RSI_14']
                clean_df = df.dropna(subset=features)
                train_df = clean_df.dropna(subset=['Target'])

                model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
                model.fit(train_df[features], train_df['Target'])

                current_price = clean_df['Close'].iloc[-1]
                predicted_price = model.predict(clean_df[features].iloc[[-1]])[0]
                diff = predicted_price - current_price
                pct = (diff / current_price) * 100
                signal = "UP / BUY" if diff > 0 else "DOWN / SELL"

                st.divider()
                st.subheader(f"Results for {ticker}")
                col1, col2 = st.columns(2)
                col1.metric("Current Price", f"{current_price:.4f}")
                col2.metric("Predicted Price", f"{predicted_price:.4f}", f"{diff:+.4f} ({pct:+.2f}%)")
                
                if diff > 0:
                    st.success(f"Signal: **{signal}**")
                else:
                    st.error(f"Signal: **{signal}**")

