import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load and preprocess data (same as before)
df = pd.read_csv('sensor_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek

# Create lag features (keeping for LSTM)
for col in ['temperature', 'vibration', 'pressure']:
    for lag in range(1, 4):
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

# Drop rolling features (LSTM handles temporal patterns inherently)
df.dropna(inplace=True)

# Normalize features
scaler = MinMaxScaler()
features = df.drop(columns=['timestamp', 'failure'])
scaled_features = scaler.fit_transform(features)

# Prepare sequences for LSTM
def create_sequences(data, targets, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(targets[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 10  # Number of timesteps to look back
X, y = create_sequences(scaled_features, df['failure'].values, SEQ_LENGTH)

# Train-test split (time-based)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTM Model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# Add class weighting (since data is imbalanced)
class_weight = {0: 1, 1: 20}  # Same as your RF weights

# Train with early stopping
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    class_weight=class_weight,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

# Evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("Accuracy:", np.mean(y_pred.flatten() == y_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Plot precision-recall curve
from sklearn.metrics import precision_recall_curve
y_probs = model.predict(X_test).flatten()
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()