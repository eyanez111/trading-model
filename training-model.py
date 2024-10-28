import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from ta.momentum import RSIIndicator

# Paths for model weights and metrics log
model_weights_path = '/home/francisco/trading-model/btc_price_model_weights.weights.h5'
metrics_log_path = '/home/francisco/trading-model/data-logs/learning_metrics.log'

# Step 1: Load and label the data for trend prediction
df = pd.read_csv('/home/francisco/trading-model/cleaned_data.csv')

# Convert timestamp to datetime object
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['timestamp'] = df['timestamp'].ffill().bfill()

# Feature Engineering
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Normalize features
df['price_usd_normalized'] = (df['price_usd'] - df['price_usd'].min()) / (df['price_usd'].max() - df['price_usd'].min())

# Calculate price change over 4-hour periods for trend labeling
df['price_change'] = df['price_usd_normalized'].pct_change(periods=4).fillna(0)

# Define thresholds for classifying trends
def classify_trend(change):
    if change > 0.005:
        return 1  # Go Up
    elif change < -0.005:
        return 2  # Go Down
    else:
        return 0  # Consolidate

df['trend'] = df['price_change'].apply(classify_trend)

# Step 2: Add RSI indicator (keep only RSI for model and alerts)
df['RSI'] = RSIIndicator(df['price_usd_normalized'], window=14).rsi()

# One-hot encode the trend column for categorical cross-entropy loss
df = pd.get_dummies(df, columns=['trend'])
df = df.fillna(0)  # Fill any remaining NaNs

# Prepare features and labels
X = df[['price_usd_normalized', 'RSI', 'hour', 'day_of_week', 'is_weekend']].values.astype('float32')
y = df[['trend_0', 'trend_1', 'trend_2']].values.astype('float32')

# Step 4: Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape input for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

# Step 5: Build the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(50, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(3, activation='softmax', kernel_regularizer=l2(0.001)))
    model.compile(optimizer=Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build the model
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_model(input_shape)

# Load only model weights, ignoring optimizer state
if os.path.exists(model_weights_path):
    print("Loading existing model weights...")
    model.load_weights(model_weights_path)
else:
    print("No existing weights found. Training from scratch.")

# Set up ModelCheckpoint and EarlyStopping callbacks
checkpoint_callback = ModelCheckpoint(filepath=model_weights_path, save_weights_only=True, save_best_only=True, verbose=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=2,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# Step 7: Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Final Validation Loss: {val_loss}")
print(f"Final Validation Accuracy: {val_accuracy}")

# Step 8: Make Predictions on the validation set
y_pred = model.predict(X_val)
pred_classes = np.argmax(y_pred, axis=1)

# Alert Logic - Check every 4 hours for alignment in model prediction and RSI signals
# Alert Logic - Model prediction first, then check RSI for alignment
for i in range(0, len(pred_classes), 4):  # Check every 4 hours
    rsi = df.iloc[i + len(X_train)]['RSI']
    model_prediction = pred_classes[i]  # Model prediction: 0 = Consolidate, 1 = Go Up, 2 = Go Down
    
    # Debugging print statements to observe model predictions and RSI values
    print(f"Index {i}: Model Prediction = {model_prediction}, RSI = {rsi}")

    # Model predicts "Go Up" (1)
    if model_prediction == 1:
        if rsi <= 35:
            print(f"Alert at index {i}: Signal to BUY (Model: Go Up, RSI {rsi} <= 35)")
        else:
            print(f"Alert at index {i}: Model predicts BUY, but RSI does not confirm (RSI {rsi})")

    # Model predicts "Go Down" (2)
    elif model_prediction == 2:
        if rsi >= 70:
            print(f"Alert at index {i}: Signal to SELL (Model: Go Down, RSI {rsi} >= 70)")
        else:
            print(f"Alert at index {i}: Model predicts SELL, but RSI does not confirm (RSI {rsi})")

    # Model predicts "Consolidate" (0)
    elif model_prediction == 0:
        if 45 <= rsi <= 65:
            print(f"Alert at index {i}: Signal for CONSOLIDATION (Model: Consolidate, RSI {rsi} between 45 and 65)")
        else:
            print(f"Alert at index {i}: Model predicts CONSOLIDATION, but RSI does not confirm (RSI {rsi})")

    # If none of the conditions are met, output "Nothing to do"
    else:
        print(f"Alert at index {i}: Nothing to do")


# Plotting loss and accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
