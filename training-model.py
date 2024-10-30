import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Paths for model weights and metrics log
model_weights_path = '/home/francisco/trading-model/btc_price_model_weights.keras'
metrics_log_path = '/home/francisco/trading-model/data-logs/learning_metrics.log'

# Load and preprocess data
df = pd.read_csv('/home/francisco/trading-model/cleaned_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['timestamp'] = df['timestamp'].ffill().bfill()

# Normalize price and calculate price change
df['price_usd_normalized'] = (df['price_usd'] - df['price_usd'].min()) / (df['price_usd'].max() - df['price_usd'].min())
df['price_change'] = df['price_usd_normalized'].pct_change(periods=4).fillna(0)
df['price_change_scaled'] = (df['price_change'] - df['price_change'].min()) / (df['price_change'].max() - df['price_change'].min())

# Define thresholds for trends
def classify_trend(change):
    if change > 0.01:
        return 1  # Go Up
    elif change < -0.01:
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

# Prepare features and labels
X = balanced_df[['price_usd_normalized', 'price_change_scaled']].values.astype('float32')
y = balanced_df[['trend_0', 'trend_1', 'trend_2']].values.astype('float32')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape input for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

# Model architecture with added LSTM layer and higher dropout
def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        Dropout(0.3),
        LSTM(50, return_sequences=False),
        Dropout(0.3),
        Dense(20, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize model
model = build_model((X_train.shape[1], X_train.shape[2]))

# Define callbacks
callbacks = [
    ModelCheckpoint(filepath=model_weights_path, save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
]

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=2,
    callbacks=callbacks
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Prediction logic - Check every 4 hours
y_pred = model.predict(X_val)
pred_classes = np.argmax(y_pred, axis=1)

# Adjust the loop to check predictions every 4 intervals at the end of the validation set
for i in range(len(pred_classes) - 8, len(pred_classes), 4):
    action = ["Consolidate", "BUY", "SELL"][pred_classes[i]]
    print(f"Alert at index {i}: Model predicts {action}")


# Plotting training loss and accuracy
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
