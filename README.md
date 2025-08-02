import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Broker login
def login_mt5():
    mt5.initialize(login=123456, server="YourBroker-Server", password="yourpassword")

# Get historical data
def get_gold_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M15, bars=500):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Feature engineering
def prepare_features(df):
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    df['rsi'] = compute_rsi(df['close'], 14)
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df = df.dropna()
    return df

def compute_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Train ML model
def train_model(df):
    features = ['ma_20', 'ma_50', 'rsi']
    X = df[features]
    y = df['target']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# Predict signal
def predict_signal(model, latest_data):
    features = ['ma_20', 'ma_50', 'rsi']
    signal = model.predict(latest_data[features])
    return signal[0]

# Place order
def place_order(symbol, order_type, volume=0.01):
    # Order types: mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": "AI Gold Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    result = mt5.order_send(request)
    return result

if __name__ == "__main__":
    login_mt5()
    df = get_gold_data()
    df = prepare_features(df)
    model = train_model(df)
    
    # Get latest data for prediction
    latest_data = df.iloc[[-1]]
    signal = predict_signal(model, latest_data)
    symbol = "XAUUSD"
    if signal == 1:
        print("Buy signal detected.")
        place_order(symbol, mt5.ORDER_TYPE_BUY, volume=0.01)
    else:
        print("Sell signal detected.")
        place_order(symbol, mt5.ORDER_TYPE_SELL, volume=0.01)
