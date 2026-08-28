import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="Macro-Sentiment Quant Predictor", layout="centered")
st.title("Neural Quant Predictor (Macro + Sentiment + DL)")

ticker = st.text_input("Enter Ticker (e.g. BTC-USD, RELIANCE.NS, NVDA, EURUSD=X):", "").strip().upper()

def get_sentiment(symbol):
    try:
        t = yf.Ticker(symbol)
        news = t.news
        if not news:
            return 0.0
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(item.get('title', ''))['compound'] for item in news]
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0

def fetch_macro_series(macro_symbol, target_index):
    try:
        macro_df = yf.download(macro_symbol, period="3y", interval="1d", progress=False)
        if macro_df.empty:
            return pd.Series(0.0, index=target_index)
        if isinstance(macro_df.columns, pd.MultiIndex):
            close_s = macro_df['Close'].iloc[:, 0]
        else:
            close_s = macro_df['Close']
        return close_s.reindex(target_index).ffill().bfill()
    except Exception:
        return pd.Series(0.0, index=target_index)

if st.button("Run Quantitative Analysis"):
    if not ticker:
        st.error("Please enter a valid ticker.")
    else:
        with st.spinner("Processing Macro regimes, News sentiment & Deep ML engines..."):
            asset = yf.Ticker(ticker)
            df = asset.history(period="3y", interval="1d", auto_adjust=False)

            if df.empty or len(df) < 120:
                st.error("Data fetch failed or insufficient trading history.")
            else:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                try:
                    current_price = float(asset.fast_info['lastPrice'])
                    if np.isnan(current_price) or current_price <= 0:
                        current_price = float(df['Close'].iloc[-1])
                except Exception:
                    current_price = float(df['Close'].iloc[-1])

                df['Macro_VIX'] = fetch_macro_series("^VIX", df.index)
                df['Macro_TNX'] = fetch_macro_series("^TNX", df.index)
                df['Macro_DXY'] = fetch_macro_series("DX-Y.NYB", df.index)

                df['Return'] = df['Close'].pct_change()
                df['Log_Return'] = np.log(df['Close'] / (df['Close'].shift(1) + 1e-9))
                
                df['SMA_20'] = df['Close'].rolling(20).mean()
                df['SMA_50'] = df['Close'].rolling(50).mean()
                df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
                df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = df['EMA_12'] - df['EMA_26']
                df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

                df['Rolling_Std_20'] = df['Close'].rolling(20).std()
                df['BB_Upper'] = df['SMA_20'] + (2 * df['Rolling_Std_20'])
                df['BB_Lower'] = df['SMA_20'] - (2 * df['Rolling_Std_20'])
                df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['SMA_20'] + 1e-9)

                delta = df['Close'].diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rs = gain / (loss + 1e-9)
                df['RSI'] = 100 - (100 / (1 + rs))

                high_low = df['High'] - df['Low']
                high_close = (df['High'] - df['Close'].shift(1)).abs()
                low_close = (df['Low'] - df['Close'].shift(1)).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['ATR_14'] = tr.rolling(14).mean()

                df['Target_Log_Return'] = np.log(df['Close'].shift(-1) / (df['Close'] + 1e-9))

                features = [
                    'Open', 'High', 'Low', 'Close', 'Volume',
                    'Return', 'Log_Return', 'Rolling_Std_20',
                    'SMA_20', 'SMA_50', 'MACD', 'MACD_Signal',
                    'BB_Width', 'RSI', 'ATR_14',
                    'Macro_VIX', 'Macro_TNX', 'Macro_DXY'
                ]

                clean_df = df.dropna(subset=features)
                train_data = clean_df.dropna(subset=['Target_Log_Return'])

                X = train_data[features]
                y = train_data['Target_Log_Return']

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                latest_raw = clean_df[features].iloc[[-1]]
                latest_scaled = scaler.transform(latest_raw)

                nn_model = MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32),
                    activation='relu',
                    solver='adam',
                    max_iter=400,
                    random_state=42
                )
                nn_model.fit(X_scaled, y)
                pred_nn = float(nn_model.predict(latest_scaled)[0])

                rf_model = RandomForestRegressor(n_estimators=120, max_depth=6, random_state=42)
                rf_model.fit(X, y)
                pred_rf = float(rf_model.predict(latest_raw)[0])

                blended_log_return = (0.6 * pred_nn) + (0.4 * pred_rf)

                live_sentiment = get_sentiment(ticker)
                sentiment_shock = live_sentiment * 0.003
                final_log_return = blended_log_return + sentiment_shock

                predicted_price = current_price * np.exp(final_log_return)
                price_diff = predicted_price - current_price
                pct_change = (price_diff / current_price) * 100

                returns_series = clean_df['Return'].dropna()
                mean_return = returns_series.mean() * 252
                volatility = returns_series.std() * np.sqrt(252)
                rf_rate = 0.05
                sharpe_ratio = (mean_return - rf_rate) / (volatility + 1e-9)

                negative_returns = returns_series[returns_series < 0]
                downside_std = negative_returns.std() * np.sqrt(252)
                sortino_ratio = (mean_return - rf_rate) / (downside_std + 1e-9)
                var_95 = (returns_series.mean() - (1.645 * returns_series.std())) * 100

                if pct_change > 0.15:
                    signal = "STRONG BUY / BULLISH"
                elif pct_change < -0.15:
                    signal = "STRONG SELL / BEARISH"
                else:
                    signal = "NEUTRAL / SIDEWAYS"

                st.divider()
                st.subheader(f"Quant Summary: {ticker}")

                c1, c2 = st.columns(2)
                c1.metric("Current Market Price", f"{current_price:.4f}")
                c2.metric("Predicted Next Close", f"{predicted_price:.4f}", f"{price_diff:+.4f} ({pct_change:+.2f}%)")

                if "BUY" in signal:
                    st.success(f"Signal: **{signal}**")
                elif "SELL" in signal:
                    st.error(f"Signal: **{signal}**")
                else:
                    st.warning(f"Signal: **{signal}**")

                st.markdown("### Model Breakdown & External Factors")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("News Sentiment", f"{live_sentiment:+.2f}")
                r2.metric("Macro VIX (Fear)", f"{clean_df['Macro_VIX'].iloc[-1]:.2f}")
                r3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                r4.metric("1-Day 95% VaR", f"{var_95:.2f}%")
