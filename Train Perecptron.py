import numpy as np
import pandas as pd
from tradingview_ta import TA_Handler, Interval
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def fetch_data(symbol, interval):
    handler = TA_Handler(symbol=symbol, screener="america", exchange="NASDAQ", interval=interval)
    history = handler.get_historical(interval)
    data = pd.DataFrame(history)
    data.set_index('timestamp', inplace=True)
    return data

def prepare_dataset(data):
    exp12 = data['close'].ewm(span=12, adjust=False).mean()
    exp26 = data['close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal_line = macd.ewm(span=9, adjust=False).mean()

    data['MACD'] = macd
    data['Signal Line'] = signal_line
    data['MACD_diff'] = data['MACD'] - data['Signal Line']
    data['Target'] = np.where(data['MACD_diff'].shift(-1) > 0, 1, 0)
    data.dropna(inplace=True)

    return data

def train_test_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Perceptron(random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, scaler

# Fetch historical data for multiple timeframes
symbol = "AAPL"
timeframes = [
    Interval.INTERVAL_5_MINUTES,
    Interval.INTERVAL_15_MINUTES,
    Interval.INTERVAL_30_MINUTES,
    Interval.INTERVAL_1_HOUR,
    Interval.INTERVAL_4_HOURS,
    Interval.INTERVAL_1_DAY
]

models = []
scalers = []
for timeframe in timeframes:
    data = fetch_data(symbol, timeframe)
    prepared_data = prepare_dataset(data)

    X = prepared_data[['MACD_diff', 'close']]
    y = prepared_data['Target']
    model, accuracy, scaler = train_test_model(X, y)

    print(f"Model accuracy for {timeframe}: {accuracy}")

    models.append((timeframe, model))
    scalers.append(scaler)

# Example input data: MACD_diff = 0.05, Close price = 150
input_data = np.array([[0.05, 150]])
min_agreement = 2

# Preprocess the input data for each timeframe
input_data_prepared = [scaler.transform(input_data) for scaler in scalers]

# Make a combined prediction using a voting system
predictions = [model.predict(input_data_prepared[idx]) for idx, (_, model) in enumerate(models)]
sum_predictions = np.sum(predictions)

if sum_predictions >= min_agreement:
    combined_prediction = 1
elif sum_predictions <= len(models) - min_agreement:
    combined_prediction = 0
else:
    combined_prediction = -1

print("Combined prediction:", combined_prediction)

