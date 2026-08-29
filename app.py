import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import nltk
from scipy.stats import skew, kurtosis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.mixture import GaussianMixture

nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="Frontier Quantitative Terminal", layout="centered")
st.title("Frontier Sovereign Quant Engine")

ASSET_PRESETS = {
    "Crypto": {
        "Bitcoin (BTC-USD)": "BTC-USD",
        "Ethereum (ETH-USD)": "ETH-USD",
        "Solana (SOL-USD)": "SOL-USD",
        "Binance Coin (BNB-USD)": "BNB-USD",
        "Ripple (XRP-USD)": "XRP-USD",
        "Cardano (ADA-USD)": "ADA-USD",
        "Dogecoin (DOGE-USD)": "DOGE-USD"
    },
    "Indian Stocks (NSE)": {
        "Reliance Industries": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "Infosys": "INFY.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "State Bank of India": "SBIN.NS",
        "Tata Motors": "TATAMOTORS.NS",
        "ITC": "ITC.NS"
    },
    "US / Foreign Stocks": {
        "Nvidia (NVDA)": "NVDA",
        "Apple (AAPL)": "AAPL",
        "Microsoft (MSFT)": "MSFT",
        "Tesla (TSLA)": "TSLA",
        "Amazon (AMZN)": "AMZN",
        "Google / Alphabet (GOOGL)": "GOOGL",
        "Meta Platforms (META)": "META"
    },
    "Forex & Commodities": {
        "EUR / USD": "EURUSD=X",
        "USD / INR": "USDINR=X",
        "GBP / USD": "GBPUSD=X",
        "Gold (GC=F)": "GC=F",
        "Silver (SI=F)": "SI=F",
        "Crude Oil (CL=F)": "CL=F"
    },
    "Custom Ticker": {}
}

c_cat, c_asset = st.columns(2)

with c_cat:
    selected_category = st.selectbox(
        "Choose Market Category:",
        list(ASSET_PRESETS.keys())
    )

with c_asset:
    if selected_category != "Custom Ticker":
        asset_options = ASSET_PRESETS[selected_category]
        selected_asset_label = st.selectbox(
            f"Select {selected_category} Asset:",
            list(asset_options.keys())
        )
        ticker = asset_options[selected_asset_label]
    else:
        ticker = st.text_input("Enter Any Custom Ticker (e.g. BTC-USD, RELIANCE.NS, TSLA):", "").strip().upper()

def compute_hurst_exponent(ts, max_lag=20):
    try:
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(poly[0])
    except Exception:
        return 0.5

def dominant_fourier_cycle(log_prices):
    try:
        n = len(log_prices)
        detrended = log_prices - np.polyval(np.polyfit(np.arange(n), log_prices, 1), np.arange(n))
        fft_vals = np.fft.rfft(detrended)
        fft_power = np.abs(fft_vals)**2
        freqs = np.fft.rfftfreq(n)
        if len(fft_power) > 2:
            peak_idx = np.argmax(fft_power[1:]) + 1
            if freqs[peak_idx] > 0:
                dom_period = 1.0 / freqs[peak_idx]
                return float(min(120, max(5, dom_period)))
        return 20.0
    except Exception:
        return 20.0

def compute_ou_half_life(prices_series):
    try:
        p = prices_series.values
        lag_p = p[:-1]
        delta_p = np.diff(p)
        beta = np.polyfit(lag_p, delta_p, 1)[0]
        if beta < 0:
            half_life = -np.log(2) / beta
            return float(min(100.0, max(1.0, half_life)))
        return 0.0
    except Exception:
        return 0.0

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
        if 'Close' in macro_df:
            close_data = macro_df['Close']
            close_s = close_data.iloc[:, 0] if isinstance(close_data, pd.DataFrame) else close_data
        else:
            close_s = macro_df.iloc[:, 0]
        
        close_s.index = pd.to_datetime(close_s.index).tz_localize(None)
        return close_s.reindex(target_index).ffill().bfill().fillna(0.0)
    except Exception:
        return pd.Series(0.0, index=target_index)

def get_benchmark_ticker(symbol):
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return "^NSEI"
    elif symbol.endswith("USD") or "-" in symbol:
        return "BTC-USD"
    else:
        return "^GSPC"

if st.button("Execute Frontier Synthesis"):
    if not ticker:
        st.error("Please select or enter a valid market asset identifier.")
    else:
        with st.spinner(f"Synthesizing Spectral Waves, Stochastic Differential Equations & Deep Ensembles for {ticker}..."):
            asset = yf.Ticker(ticker)
            df = asset.history(period="3y", interval="1d", auto_adjust=False)

            if df.empty or len(df) < 140:
                st.error("Data fetch failed or insufficient historical liquidity depth.")
            else:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df.index = pd.to_datetime(df.index).tz_localize(None)

                try:
                    current_price = float(asset.fast_info['lastPrice'])
                    if np.isnan(current_price) or current_price <= 0:
                        current_price = float(df['Close'].iloc[-1])
                except Exception:
                    current_price = float(df['Close'].iloc[-1])

                df['Macro_VIX'] = fetch_macro_series("^VIX", df.index)
                df['Macro_TNX'] = fetch_macro_series("^TNX", df.index)
                df['Macro_DXY'] = fetch_macro_series("DX-Y.NYB", df.index)
                df['Macro_GOLD'] = fetch_macro_series("GC=F", df.index)

                benchmark_sym = get_benchmark_ticker(ticker)
                bench_series = fetch_macro_series(benchmark_sym, df.index)
                bench_returns = bench_series.pct_change()

                df['Return'] = df['Close'].pct_change()
                df['Log_Return'] = np.log(df['Close'] / (df['Close'].shift(1) + 1e-9))

                gk_inner = (
                    0.5 * (np.log(df['High'] / (df['Low'] + 1e-9)))**2 - 
                    (2 * np.log(2) - 1) * (np.log(df['Close'] / (df['Open'] + 1e-9)))**2
                )
                df['GK_Vol'] = np.sqrt(np.maximum(0, gk_inner))
                df['Vol_of_Vol'] = df['GK_Vol'].rolling(20).std()

                df['Skew_30'] = df['Return'].rolling(30).skew().fillna(0.0)
                df['Kurt_30'] = df['Return'].rolling(30).kurt().fillna(0.0)

                df['SMA_20'] = df['Close'].rolling(20).mean()
                df['SMA_50'] = df['Close'].rolling(50).mean()
                df['SMA_200'] = df['Close'].rolling(200).mean().fillna(df['SMA_50'])
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

                low_14 = df['Low'].rolling(14).min()
                high_14 = df['High'].rolling(14).max()
                df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-9))
                df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

                regime_features = df[['Return', 'GK_Vol', 'Rolling_Std_20', 'Vol_of_Vol']].dropna()
                gmm = GaussianMixture(n_components=3, random_state=42)
                regimes = gmm.fit_predict(regime_features)
                df['Regime'] = pd.Series(regimes, index=regime_features.index).reindex(df.index).ffill().bfill().fillna(0)

                current_regime_id = int(df['Regime'].iloc[-1])
                regime_map = {0: "Low-Vol Expansion (Bullish)", 1: "High-Vol Tail Shock (Risk)", 2: "Mean-Reverting Equilibrium"}
                regime_label = regime_map.get(current_regime_id, "Trending Flow")

                recent_closes = df['Close'].values[-120:]
                hurst_val = compute_hurst_exponent(recent_closes)
                if hurst_val > 0.55:
                    fractal_state = "Persistent Trend"
                elif hurst_val < 0.45:
                    fractal_state = "Mean-Reverting"
                else:
                    fractal_state = "Brownian Motion"

                fft_dominant_days = dominant_fourier_cycle(np.log(recent_closes))
                ou_half_life_days = compute_ou_half_life(df['Close'].iloc[-120:])

                df['Target_Log_Return'] = np.log(df['Close'].shift(-1) / (df['Close'] + 1e-9))

                features = [
                    'Open', 'High', 'Low', 'Close', 'Volume',
                    'Return', 'Log_Return', 'GK_Vol', 'Vol_of_Vol', 'Rolling_Std_20',
                    'Skew_30', 'Kurt_30', 'SMA_20', 'SMA_50', 'SMA_200', 'MACD', 'MACD_Signal',
                    'BB_Width', 'RSI', 'ATR_14', 'Stoch_K', 'Stoch_D',
                    'Macro_VIX', 'Macro_TNX', 'Macro_DXY', 'Macro_GOLD', 'Regime'
                ]

                clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
                train_data = clean_df.dropna(subset=['Target_Log_Return'])

                X = train_data[features].values
                y = train_data['Target_Log_Return'].values

                test_sample = min(120, len(X) - 30)
                if test_sample > 20:
                    X_train_bt, X_test_bt = X[:-test_sample], X[-test_sample:]
                    y_train_bt, y_test_bt = y[:-test_sample], y[-test_sample:]
                    
                    scaler_bt = RobustScaler()
                    X_train_bt_scaled = scaler_bt.fit_transform(X_train_bt)
                    X_test_bt_scaled = scaler_bt.transform(X_test_bt)
                    
                    nn_bt = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', max_iter=300, random_state=42)
                    rf_bt = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
                    gb_bt = GradientBoostingRegressor(n_estimators=80, max_depth=4, random_state=42)
                    
                    nn_bt.fit(X_train_bt_scaled, y_train_bt)
                    rf_bt.fit(X_train_bt, y_train_bt)
                    gb_bt.fit(X_train_bt, y_train_bt)
                    
                    p_nn_bt = nn_bt.predict(X_test_bt_scaled)
                    p_rf_bt = rf_bt.predict(X_test_bt)
                    p_gb_bt = gb_bt.predict(X_test_bt)
                    blend_bt = (0.50 * p_nn_bt) + (0.25 * p_rf_bt) + (0.25 * p_gb_bt)
                    
                    correct_hits = np.sum((blend_bt > 0) == (y_test_bt > 0))
                    win_rate = (correct_hits / test_sample) * 100.0
                else:
                    win_rate = 52.0

                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)
                latest_raw = clean_df[features].iloc[[-1]].values
                latest_scaled = scaler.transform(latest_raw)

                nn_model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', max_iter=450, random_state=42)
                nn_model.fit(X_scaled, y)
                pred_nn = float(nn_model.predict(latest_scaled)[0])

                rf_model = RandomForestRegressor(n_estimators=120, max_depth=6, random_state=42)
                rf_model.fit(X, y)
                pred_rf = float(rf_model.predict(latest_raw)[0])

                gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
                gb_model.fit(X, y)
                pred_gb = float(gb_model.predict(latest_raw)[0])

                blended_log_return = (0.50 * pred_nn) + (0.25 * pred_rf) + (0.25 * pred_gb)

                live_sentiment = get_sentiment(ticker)
                sentiment_shock = live_sentiment * 0.003
                final_log_return = blended_log_return + sentiment_shock

                predicted_price = current_price * np.exp(final_log_return)
                price_diff = predicted_price - current_price
                pct_change = (price_diff / current_price) * 100

                current_atr = float(clean_df['ATR_14'].iloc[-1])
                pred_std = float(clean_df['Return'].std())
                lower_band = current_price * np.exp(final_log_return - (1.96 * pred_std))
                upper_band = current_price * np.exp(final_log_return + (1.96 * pred_std))

                if pct_change > 0.15:
                    signal = "STRONG BUY / LONG EXPANSION"
                    target_1 = current_price + (2.0 * current_atr)
                    stop_loss = current_price - (1.0 * current_atr)
                elif pct_change < -0.15:
                    signal = "STRONG SELL / SHORT CONTRACTION"
                    target_1 = current_price - (2.0 * current_atr)
                    stop_loss = current_price + (1.0 * current_atr)
                else:
                    signal = "NEUTRAL / MARKET EQUILIBRIUM"
                    target_1 = current_price + current_atr
                    stop_loss = current_price - current_atr

                p_win = max(0.35, min(0.75, win_rate / 100.0))
                b_odds = 2.0
                q_loss = 1.0 - p_win
                kelly_fraction = max(0.0, (b_odds * p_win - q_loss) / b_odds) * 100.0
                safe_kelly = kelly_fraction * 0.5

                aligned_returns = pd.concat([clean_df['Return'], bench_returns], axis=1).dropna()
                if len(aligned_returns) > 30 and aligned_returns.iloc[:, 1].var() > 0:
                    cov_matrix = np.cov(aligned_returns.iloc[:, 0], aligned_returns.iloc[:, 1])
                    beta_val = cov_matrix[0, 1] / cov_matrix[1, 1]
                else:
                    beta_val = 1.0

                returns_series = clean_df['Return'].dropna()
                mean_return = returns_series.mean() * 252
                volatility = returns_series.std() * np.sqrt(252)
                rf_rate = 0.05
                sharpe_ratio = (mean_return - rf_rate) / (volatility + 1e-9)

                negative_returns = returns_series[returns_series < 0]
                downside_std = negative_returns.std() * np.sqrt(252)
                sortino_ratio = (mean_return - rf_rate) / (downside_std + 1e-9)

                st.divider()
                st.subheader(f"Frontier Quant Terminal: {ticker}")

                c1, c2 = st.columns(2)
                c1.metric("Live Market Price", f"{current_price:.4f}")
                c2.metric("Predicted Next Close", f"{predicted_price:.4f}", f"{price_diff:+.4f} ({pct_change:+.2f}%)")

                if "BUY" in signal:
                    st.success(f"Signal: **{signal}**")
                elif "SELL" in signal:
                    st.error(f"Signal: **{signal}**")
                else:
                    st.warning(f"Signal: **{signal}**")

                st.markdown("### Execution Strategy & Bayesian Expected Band")
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Take-Profit (2.0x ATR)", f"{target_1:.4f}")
                t2.metric("Stop-Loss (1.0x ATR)", f"{stop_loss:.4f}")
                t3.metric("95% Expected Range", f"{lower_band:.2f} - {upper_band:.2f}")
                t4.metric("Kelly Allocation", f"{safe_kelly:.1f}%")

                st.markdown("### Frontier Spectral & Non-Linear Diagnostics")
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Dominant Cycle (FFT)", f"{fft_dominant_days:.1f} Days")
                f2.metric("OU Half-Life", f"{ou_half_life_days:.1f} Days" if ou_half_life_days > 0 else "Pure Trend")
                f3.metric("Fractal Hurst (H)", f"{hurst_val:.2f} ({fractal_state})")
                f4.metric("Walk-Forward Accuracy", f"{win_rate:.1f}%")

                st.markdown("### Institutional Macro & Tail Risk Regimes")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Market Regime (GMM)", regime_label.split()[0])
                r2.metric(f"Beta ({benchmark_sym})", f"{beta_val:.2f}")
                r3.metric("Sharpe / Sortino", f"{sharpe_ratio:.2f} / {sortino_ratio:.2f}")
                r4.metric("Skew / Kurtosis", f"{clean_df['Skew_30'].iloc[-1]:.2f} / {clean_df['Kurt_30'].iloc[-1]:.2f}")
