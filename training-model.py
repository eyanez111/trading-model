import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Set TensorFlow logging level to only show errors
tf.get_logger().setLevel('ERROR')

# Set random seeds
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Load and preprocess data
df = pd.read_csv('/home/francisco/trading-model/cleaned_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.tz_localize('UTC')
df['timestamp'] = df['timestamp'].ffill().bfill()

# Feature engineering
df['price_usd_normalized'] = (df['price_usd'] - df['price_usd'].min()) / (df['price_usd'].max() - df['price_usd'].min())
df['price_change'] = df['price_usd_normalized'].pct_change(periods=4).fillna(0)
df['price_change_scaled'] = (df['price_change'] - df['price_change'].min()) / (df['price_change'].max() - df['price_change'].min())
df['volatility'] = df['price_usd'].rolling(window=10).std().fillna(0)
df['moving_avg_3'] = df['price_usd'].rolling(window=3).mean().fillna(0)
df['moving_avg_5'] = df['price_usd'].rolling(window=5).mean().fillna(0)
df['moving_avg_10'] = df['price_usd'].rolling(window=10).mean().fillna(0)

# Define thresholds for trends
def classify_trend(change):
    if change > 0.02:
        return 1  # Go Up
    elif change < -0.02:
        return 2  # Go Down
    else:
        return 0  # Consolidate

df['trend'] = df['price_change'].apply(classify_trend)
df = pd.get_dummies(df, columns=['trend']).fillna(0)

# Balance the data samples for each trend
min_samples = min(df['trend_0'].sum(), df['trend_1'].sum(), df['trend_2'].sum())
balanced_df = pd.concat([
    df[df['trend_0'] == 1].sample(n=min_samples),
    df[df['trend_1'] == 1].sample(n=min_samples),
    df[df['trend_2'] == 1].sample(n=min_samples)
])

# Validate the class balance after sampling
print("Class distribution after balancing:")
print("Consolidate (trend_0):", balanced_df['trend_0'].sum())
print("Go Up (trend_1):", balanced_df['trend_1'].sum())
print("Go Down (trend_2):", balanced_df['trend_2'].sum())

# Prepare features and labels with 5 timesteps
X = balanced_df[['price_usd_normalized', 'price_change_scaled', 'volatility', 'moving_avg_3', 'moving_avg_5', 'moving_avg_10']].values.astype('float32')
y = balanced_df[['trend_0', 'trend_1', 'trend_2']].values.astype('float32')

sequence_length = 5
X = np.array([X[i - sequence_length:i] for i in range(sequence_length, len(X))])
y = y[sequence_length:]  # Align y with X

# Flatten X for Random Forest
X_rf = X.reshape((X.shape[0], -1))  # Flatten sequence data
y_rf = np.argmax(y, axis=1)         # Convert one-hot encoding to single labels

# Define LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(20, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=5e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define GRU model
def build_gru_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True),
        Dropout(0.3),
        GRU(64, return_sequences=True),
        Dropout(0.3),
        GRU(32),
        Dropout(0.3),
        Dense(20, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=5e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Prediction function
def make_predictions(model, X_val):
    return model.predict(X_val, batch_size=32)

# Implement K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
fold = 1
validation_accuracies = []

# Initialize counters for each trend
total_buy = 0
total_sell = 0
total_consolidate = 0

for train_index, val_index in kf.split(X):
    print(f"Starting fold {fold}")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    X_rf_train, X_rf_val = X_rf[train_index], X_rf[val_index]
    y_rf_train, y_rf_val = y_rf[train_index], y_rf[val_index]
    
    # Train models and make predictions as before (LSTM, GRU, RandomForest)
    lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=2)
    
    gru_model = build_gru_model((X_train.shape[1], X_train.shape[2]))
    gru_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=2)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf_model.fit(X_rf_train, y_rf_train)
    
    # Make ensemble predictions
    lstm_pred = make_predictions(lstm_model, X_val)
    gru_pred = make_predictions(gru_model, X_val)
    rf_pred = rf_model.predict_proba(X_rf_val)
    ensemble_pred = (lstm_pred + gru_pred + rf_pred) / 3
    ensemble_classes = np.argmax(ensemble_pred, axis=1)

    # Count each trend across the fold
    total_buy += np.sum(ensemble_classes == 1)
    total_sell += np.sum(ensemble_classes == 2)
    total_consolidate += np.sum(ensemble_classes == 0)

    # Generate 4-hour interval alerts based on calculations
    trend_labels = {0: "Consolidate", 1: "BUY", 2: "SELL"}
    i = len(ensemble_classes) - 1  # Last index
    if i >= 4:  # Ensure there is a previous cycle
        previous_trend = trend_labels[ensemble_classes[i - 4]]
        current_trend = trend_labels[ensemble_classes[i]]

        # Print only alerts for 4-hour interval predictions
        print(f"Previous Alert: {previous_trend}")
        print(f"Current Alert: {current_trend}\n")

    # Calculate fold accuracy
    val_accuracy = np.mean(ensemble_classes == np.argmax(y_val, axis=1))
    print(f"Fold {fold} - Ensemble Validation Accuracy: {val_accuracy}")
    validation_accuracies.append(val_accuracy)
    fold += 1

# Final average accuracy across folds
avg_val_accuracy = np.mean(validation_accuracies)
print(f"\nAverage Ensemble Validation Accuracy across folds: {avg_val_accuracy}")

# Summary of predictions across all folds
print("\nSummary of Predictions Across All Folds:")
print(f"Total BUY predictions: {total_buy}")
print(f"Total SELL predictions: {total_sell}")
print(f"Total Consolidate predictions: {total_consolidate}")
