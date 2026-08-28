hereimport streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Deep Learning Market Predictor", layout="centered")
st.title("Neural Quant Predictor (DL + ML)")

ticker = st.text_input("Enter Ticker (e.g. BTC-USD, RELIANCE.NS, NVDA, EURUSD=X):", "").strip().upper()

class MarketLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(MarketLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

if st.button("Run Deep Learning Analysis"):
    if not ticker:
        st.error("Please enter a valid ticker.")
    else:
        with st.spinner("Training Deep Neural Network & Quant Engines..."):
            asset = yf.Ticker(ticker)
            df = asset.history(period="4y", interval="1d", auto_adjust=False)

            if df.empty or len(df) < 200:
                st.error("Data fetch failed or insufficient trading history.")
            else:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                try:
                    current_price = float(asset.fast_info['lastPrice'])
                    if np.isnan(current_price) or current_price == 0:
                        current_price = float(df['Close'].iloc[-1])
                except Exception:
                    current_price = float(df['Close'].iloc[-1])

                df['Return'] = df['Close'].pct_change()
                df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
                df['SMA_20'] = df['Close'].rolling(20).mean()
                df['SMA_50'] = df['Close'].rolling(50).mean()
                df['Rolling_Std_20'] = df['Close'].rolling(20).std()

                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-9)
                df['RSI'] = 100 - (100 / (1 + rs))

                df['Target_Log_Return'] = df['Log_Return'].shift(-1)

                features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Log_Return', 'SMA_20', 'SMA_50', 'Rolling_Std_20', 'RSI']
                clean_df = df.dropna(subset=features)
                train_data = clean_df.dropna(subset=['Target_Log_Return'])

                scaler = MinMaxScaler()
                scaled_features = scaler.fit_transform(clean_df[features])

                seq_length = 30
                X_seq, y_seq = [], []
                target_vals = train_data['Target_Log_Return'].values

                for i in range(len(target_vals) - seq_length):
                    X_seq.append(scaled_features[i:i+seq_length])
                    y_seq.append(target_vals[i+seq_length])

                X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
                y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32).unsqueeze(1)

                lstm_model = MarketLSTM(input_dim=len(features), hidden_dim=64, num_layers=2)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)

                lstm_model.train()
                for epoch in range(40):
                    optimizer.zero_grad()
                    outputs = lstm_model(X_seq)
                    loss = criterion(outputs, y_seq)
                    loss.backward()
                    optimizer.step()

                lstm_model.eval()
                latest_seq = torch.tensor(scaled_features[-seq_length:], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    pred_dl = lstm_model(latest_seq).item()

                rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
                rf_model.fit(train_data[features], train_data['Target_Log_Return'])
                pred_ml = rf_model.predict(clean_df[features].iloc[[-1]])[0]

                blended_log_return = (0.6 * pred_dl) + (0.4 * pred_ml)
                predicted_price = current_price * np.exp(blended_log_return)
                price_diff = predicted_price - current_price
                pct_change = (price_diff / current_price) * 100

                returns_series = clean_df['Return'].dropna()
                mean_return = returns_series.mean() * 252
                volatility = returns_series.std() * np.sqrt(252)
                sharpe_ratio = (mean_return - 0.05) / (volatility + 1e-9)

                negative_returns = returns_series[returns_series < 0]
                downside_std = negative_returns.std() * np.sqrt(252)
                sortino_ratio = (mean_return - 0.05) / (downside_std + 1e-9)
                var_95 = (returns_series.mean() - (1.645 * returns_series.std())) * 100

                if pct_change > 0.15:
                    signal = "STRONG BUY / BULLISH"
                elif pct_change < -0.15:
                    signal = "STRONG SELL / BEARISH"
                else:
                    signal = "NEUTRAL / SIDEWAYS"

                st.divider()
                st.subheader(f"Deep Neural Prediction: {ticker}")

                c1, c2 = st.columns(2)
                c1.metric("Current Market Price", f"{current_price:.4f}")
                c2.metric("Predicted Next Close", f"{predicted_price:.4f}", f"{price_diff:+.4f} ({pct_change:+.2f}%)")

                if "BUY" in signal:
                    st.success(f"Signal: **{signal}**")
                elif "SELL" in signal:
                    st.error(f"Signal: **{signal}**")
                else:
                    st.warning(f"Signal: **{signal}**")

                st.markdown("### Model Breakdown & Risk Metrics")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("LSTM (Deep Learning)", f"{pred_dl * 100:+.2f}%")
                r2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                r3.metric("Annual Volatility", f"{volatility * 100:.1f}%")
                r4.metric("1-Day 95% VaR", f"{var_95:.2f}%")
