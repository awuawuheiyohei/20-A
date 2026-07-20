🧩 Step 1: Data Preparation
English Prompt
Collect historical A-share (Shanghai/Shenzhen) stock price data, including OHLC (open, high, low, close) and volume.
Include macroeconomic indicators (GDP growth, CPI, interest rates) and sector-specific indexes.
Clean and normalize data, handle missing values, and align time series.

---

📊 Step 2: Feature Engineering
English Prompt
Generate technical indicators (moving averages, RSI, MACD, Bollinger Bands).
Add lagged features and rolling statistics to capture momentum and volatility.
Encode categorical events (earnings reports, policy announcements).

---

🤖 Step 3: Model Selection
English Prompt
Train baseline models (linear regression, random forest) for benchmark comparison.
Experiment with deep learning models (LSTM, Transformer) for sequence prediction.
Evaluate models using metrics such as RMSE, MAPE, and directional accuracy.

---

⚖️ Step 4: Risk & Validation
English Prompt
Apply walk-forward validation to simulate real trading conditions.
Test robustness under different market regimes (bull, bear, sideways).
Document limitations: prediction is probabilistic, not guaranteed.

---

📈 Step 5: Strategy & Deployment
English Prompt
Integrate prediction outputs into a simple trading strategy (e.g., buy if probability of upward move > 60%).
Backtest strategy on historical data with transaction costs.
Deploy model with scheduled retraining and monitoring dashboards.