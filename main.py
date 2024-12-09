import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import datetime

today = datetime.datetime.today().strftime('%Y-%m-%d')


ticker = 'NVDA'
data = yf.download(ticker, start='2022-10-07', end=today)
df = data[['Open']]
df = df.dropna() 
df.index.name = 'Date' 
print(df.head(5))
print(df.describe())
plt.plot(df.index, df['Open'])
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title(f'{ticker} Open Price Over Time')
plt.show()

df1 = df['Open']
scaler = MinMaxScaler()
df2 = scaler.fit_transform(df1.values.reshape(-1, 1))
df2 = pd.DataFrame(df2)
df2 = df2[0]

def df_to_X_y(df, window_size=15):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)

WINDOW_SIZE = 15
X1, y1 = df_to_X_y(df2, WINDOW_SIZE)

model1 = models.Sequential()
model1.add(layers.InputLayer((WINDOW_SIZE, 1)))
model1.add(layers.LSTM(64))
model1.add(layers.Dense(8, activation='relu'))
model1.add(layers.Dense(1, activation='linear'))
model1.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=0.0001))

split_index_train = int(len(df2) * 0.7)  
split_index_val = int(len(df2) * 0.9)   
X_train, y_train = X1[:split_index_train], y1[:split_index_train]
X_val, y_val = X1[split_index_train:split_index_val], y1[split_index_train:split_index_val]
X_test, y_test = X1[split_index_val:], y1[split_index_val:]
model1.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

train_predictions = model1.predict(X_train).flatten()
train_mae = mean_absolute_error(y_train, train_predictions)
print(f"Training MAE: {train_mae}")

val_predictions = model1.predict(X_val).flatten()
val_mae = mean_absolute_error(y_val, val_predictions)
print(f"Validation MAE: {val_mae}")

test_predictions = model1.predict(X_test).flatten()
test_mae = mean_absolute_error(y_test, test_predictions)
print(f"Test MAE: {test_mae}")

train_predictions = model1.predict(X1).flatten()
train_dates = df.index[WINDOW_SIZE:len(df2)]
train_results = pd.DataFrame(data={'Date': train_dates, 'Train Predictions': train_predictions, 'Actuals': y1.flatten()})
print(train_results)

num_future_steps = 30
last_date = pd.to_datetime(df.index[-1])
future_dates = pd.date_range(last_date + pd.DateOffset(days=1), periods=num_future_steps)
print(future_dates)

future_predictions = []
recent_data = X1[-1].reshape(1, WINDOW_SIZE, 1)
for _ in range(num_future_steps):
    prediction = model1.predict(recent_data)[0][0]
    future_predictions.append(prediction)
    prediction = np.array([prediction])
    recent_data = np.append(recent_data[:, 1:, :], [[prediction]], axis=1)

future_results = pd.DataFrame(data={'Date': future_dates, 'Predictions': future_predictions})
print(future_results)

def inverse_transform_dataframe(df, scaler):
    df_inverse = df.copy()
    for col in df.columns:
        if col != 'Date':
            df_inverse[col] = scaler.inverse_transform(df[col].values.reshape(-1, 1)).flatten()
    return df_inverse

train_results = inverse_transform_dataframe(train_results, scaler)
future_results = inverse_transform_dataframe(future_results, scaler)


combined_results = pd.concat([train_results, future_results], axis=0)

combined_results.to_csv("combined_results.csv")
future_results.to_csv("future_results.csv")

print("Results saved to CSV.")
