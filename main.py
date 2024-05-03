# Importing necessary libraries
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import yfinance as yf

# Taking stock data using yfinance library
ticker = 'ASELS.IS'
data = yf.download(ticker, start='2016-01-01', end='2024-01-05')

# Close column of stock data
plt.figure(figsize=(16, 8))
plt.title('Close Price Data')
plt.plot(data['Close'])
plt.xlabel('Date', fontsize=16)
plt.ylabel('TL', fontsize=16)
plt.show()

# Extracting only the 'Close' column for further processing
dataC = data.filter(['Close'])
dataArray = dataC.values

# Determining the number of rows for training (80% of the total data in this case)
training_data_len = math.ceil(len(dataArray) * .8)

# Scaling the data to bring all values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataArray)

# Preparing training dataset
training_data = scaled_data[0:training_data_len, :]

# Initializing training datasets
x_train, y_train = [], []

# Populating training datasets (Using past 60 values to predict the next one)
for i in range(60, len(training_data)):
    x_train.append(training_data[i - 60:i, 0])
    y_train.append(training_data[i, 0])

# Converting lists to numpy arrays for LSTM model compatibility
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping the x_train dataset to make it compatible with LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Defining the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
# model.add(LSTM(50))  # optional
model.add(Dense(25))
model.add(Dense(1))

# Define early stopping callback
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Compiling the model with 'adam' optimizer and mean squared error as loss function
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model using the training data
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Preparing the test dataset
testing_data = scaled_data[training_data_len - 60:, :]
x_test, y_test = [], dataArray[training_data_len:, :]
for i in range(60, len(testing_data)):
    x_test.append(testing_data[i - 60:i, 0])

# Reshaping and converting the test dataset
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predicting using the trained model
prediction = model.predict(x_test)
prediction = scaler.inverse_transform(prediction)  # Unscaling the predicted data

# Calculating the RMSE (Root Mean Squared Error)
error = np.sqrt(np.mean(((prediction - y_test) ** 2)))
print("RMSE: ", error)

# Plotting the actual vs predicted values
train = data[:training_data_len]
valid = data[training_data_len:]
valid.loc[:, 'Predictions'] = prediction
plt.figure(figsize=(14, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=16)
plt.ylabel('Close Price Tl', fontsize=16)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Values', 'Predictions'], loc='lower right')
plt.show()

# Printing actual vs predicted prices
print(valid)

# Predicting the stock price for the next day using last 60 days' data
new_data = data.filter(['Close'])
last_sixtyDays = new_data[-60:].values
scaled_sixty = scaler.transform(last_sixtyDays)
X_test = [scaled_sixty]
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
prediction_price = model.predict(X_test)
prediction_price = scaler.inverse_transform(prediction_price)
print("Prediction:", prediction_price)

# Fetching the actual stock price for the next day to compare with our prediction
aselsan = yf.download(ticker, start='2024-01-05', end='2024-01-06')
print("Actual Price:", aselsan['Close'])
