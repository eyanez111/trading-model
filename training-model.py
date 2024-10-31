import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

# Set random seeds
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Load and preprocess data
df = pd.read_csv('/home/francisco/trading-model/cleaned_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
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

# Prepare features and labels with 5 timesteps
X = balanced_df[['price_usd_normalized', 'price_change_scaled', 'volatility', 'moving_avg_3', 'moving_avg_5', 'moving_avg_10']].values.astype('float32')
y = balanced_df[['trend_0', 'trend_1', 'trend_2']].values.astype('float32')

sequence_length = 5
X = np.array([X[i - sequence_length:i] for i in range(sequence_length, len(X))])
y = y[sequence_length:]  # Align y with X

# Build a more complex model
def build_model(input_shape):
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

# Implement K-Fold Cross-Validation with consistent seed and modified model
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
fold = 1
validation_accuracies = []

for train_index, val_index in kf.split(X):
    print(f"Starting fold {fold}")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    model_weights_path = f'/home/francisco/trading-model/btc_price_model_weights_fold_{fold}.keras'
    
    callbacks = [
        ModelCheckpoint(filepath=model_weights_path, save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)
    ]
    
    model = build_model((X_train.shape[1], X_train.shape[2]))
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=2,
        callbacks=callbacks
    )
    
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Fold {fold} - Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
    validation_accuracies.append(val_accuracy)
    fold += 1

# Final average accuracy
avg_val_accuracy = np.mean(validation_accuracies)
print(f"Average Validation Accuracy across folds: {avg_val_accuracy}")

# Predict final validation data
y_pred = model.predict(X_val)
pred_classes = np.argmax(y_pred, axis=1)
for i in range(len(pred_classes) - 8, len(pred_classes), 4):
    action = ["Consolidate", "BUY", "SELL"][pred_classes[i]]
    print(f"Alert at index {i}: Model predicts {action}")

# Plot loss and accuracy
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
