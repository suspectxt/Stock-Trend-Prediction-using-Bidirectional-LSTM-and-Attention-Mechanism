import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Attention, Concatenate
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
import ta  # Technical Analysis library
import pickle
import random
import tensorflow as tf

# Set seeds for reproducibility
seed = 35
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Fetch data
data = pd.read_csv('data.csv')
data['Date'] = pd.to_datetime(data['Date'])
print("Data fetched successfully.", len(data))
data.set_index('Date', inplace=True)
data = data[['Close', 'Volume']]  # Include Volume for A/D calculation
data = data.resample('M').ffill()

# Add technical indicators
data['EMA_7'] = ta.trend.ema_indicator(data['Close'], window=7)
data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
data['EMA_111'] = ta.trend.ema_indicator(data['Close'], window=111)
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

# Calculate the Accumulation/Distribution Line (A/D)
data['AD'] = ta.volume.acc_dist_index(high=data['Close'], low=data['Close'], close=data['Close'], volume=data['Volume'])

# Drop NA values after indicator calculation
data = data.dropna()

# Create sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length].values)
        y.append(data.iloc[i+seq_length].values[0])  # Predict the 'Close' value
    return np.array(X), np.array(y)

seq_length = 30  # Adjust as needed

# Build the model
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    lstm_out = Dropout(0.2)(lstm_out)
    attention = Attention()([lstm_out, lstm_out])
    lstm_out = Concatenate()([lstm_out, attention])
    lstm_out = Bidirectional(LSTM(64, return_sequences=False))(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    outputs = Dense(1)(lstm_out)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Rolling window backtesting
print("Starting rolling-window backtesting...")

initial_train_size = int(0.7 * len(data))
test_size = 3  # Predict 3 months at a time

all_actual = []
all_predictions = []
all_dates = []

for start in range(initial_train_size, len(data) - test_size, test_size):
    end = start + test_size

    print(f"Training from 0 to {start} and predicting from {start} to {end}...")

    # Prepare training data
    train_data = data[:start-1]
    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    X_train, y_train = create_sequences(pd.DataFrame(scaled_train_data, columns=train_data.columns), seq_length)

    # Prepare test data
    test_data = data[start-seq_length:end]

    # Check for enough data points for prediction
    scaled_test_data = scaler.transform(test_data)
    X_test, y_test = create_sequences(pd.DataFrame(scaled_test_data, columns=test_data.columns), seq_length)

    # Build and train the model
    model = build_model((seq_length, scaled_train_data.shape[1]))  # Include all features in input shape
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=24, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

    # Make predictions (with the fix for true predictions)
    last_sequence = X_test[0]
    predictions = []

    for i in range(test_size):
        next_pred = model.predict(last_sequence.reshape(1, seq_length, scaled_train_data.shape[1]))[0, 0]
        predictions.append(next_pred)
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1, :-1] = last_sequence[-2, 1:]  # Shift the features except the last one
        last_sequence[-1, -1] = next_pred  # Update with the next prediction

    # Combine predictions into a single value for the 3-month period
    combined_prediction = np.mean(predictions)

    # Invert scaling
    combined_prediction = scaler.inverse_transform(np.concatenate([np.zeros((1, scaled_train_data.shape[1]-1)), np.array([[combined_prediction]])], axis=1))[:, -1].flatten()[0]  # Inverse transform only the 'Close' value
    actual_value = np.mean(y_test)  # Actual value is the average of the next 3 months
    actual_value = scaler.inverse_transform(np.concatenate([np.zeros((1, scaled_train_data.shape[1]-1)), np.array([[actual_value]])], axis=1))[:, -1].flatten()[0]  # Inverse transform only the 'Close' value

    # Store results
    all_actual.append(actual_value)
    all_predictions.append(combined_prediction)
    all_dates.append(data.index[start])

    print(f"Predictions made for period {start} to {end}.")

print("Rolling-window backtesting completed.")

model_file = f'Ema_{1}_{2}_{3}_model.pkl'
with open(model_file, 'wb') as file:
  pickle.dump(model, file)

# Combine actual and predicted values into a DataFrame for easier manipulation
print("Combining actual and predicted values...")
result_df = pd.DataFrame({
    'Date': all_dates,
    'Actual': all_actual,
    'Predicted': all_predictions
})

# Calculate monthly returns
print("Calculating monthly returns...")
result_df.set_index('Date', inplace=True)
monthly_returns = result_df.pct_change()

# Calculate the number of correctly predicted negative months
correct_negative_predictions = sum((monthly_returns['Actual'] < 0) & (monthly_returns['Predicted'] < 0))
total_negative_months = sum(monthly_returns['Actual'] < 0)

# Print the results
print(f"Correctly predicted negative months: {correct_negative_predictions}")
print(f"Total negative months: {total_negative_months}")
print(f"Accuracy in predicting negative months: {correct_negative_predictions / total_negative_months:.2%}")

# Save the results to a CSV file
print("Saving results to CSV files...")
result_df.to_csv('predictions_and_actuals.csv')
monthly_returns.to_csv('monthly_returns.csv')

print("Results saved to 'predictions_and_actuals.csv' and 'monthly_returns.csv'.")