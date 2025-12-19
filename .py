# ==============================
# STOCK MARKET PREDICTION + BACKTESTING
# ==============================

# pip install yfinance pandas numpy matplotlib scikit-learn ta

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ------------------------------
# 1. Download Historical Data
# ------------------------------
stock_symbol = "AAPL"

data = yf.download(
    stock_symbol,
    start="2018-01-01",
    end="2024-01-01",
    auto_adjust=False
)

# FIX for Google Colab (MultiIndex columns)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# ------------------------------
# 2. Feature Engineering
# ------------------------------
data['SMA_20'] = SMAIndicator(data['Close'], window=20).sma_indicator()
data['SMA_50'] = SMAIndicator(data['Close'], window=50).sma_indicator()
data['RSI'] = RSIIndicator(data['Close'], window=14).rsi()

macd = MACD(data['Close'])
data['MACD'] = macd.macd()
data['MACD_signal'] = macd.macd_signal()

data.dropna(inplace=True)

# ------------------------------
# 3. Target Variable
# ------------------------------
data['Tomorrow'] = data['Close'].shift(-1)
data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)
data.dropna(inplace=True)

# ------------------------------
# 4. Features & Labels
# ------------------------------
features = [
    'Close',
    'SMA_20',
    'SMA_50',
    'RSI',
    'MACD',
    'MACD_signal'
]

X = data[features]
y = data['Target']

# ------------------------------
# 5. Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ------------------------------
# 6. Train Model
# ------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ------------------------------
# 7. Evaluation
# ------------------------------
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# ------------------------------
# 8. Latest Signal
# ------------------------------
latest_data = X.iloc[-1:]
signal = model.predict(latest_data)

print("\n📈 BUY Signal" if signal[0] == 1 else "\n📉 SELL Signal")

# ------------------------------
# 9. Prediction for Backtesting
# ------------------------------
data['Prediction'] = model.predict(X)

# ------------------------------
# 10. Backtesting
# ------------------------------
initial_capital = 100000

data['Market_Return'] = data['Close'].pct_change()

