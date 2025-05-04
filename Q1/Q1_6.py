import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_curve, 
                             roc_auc_score, precision_recall_curve, ConfusionMatrixDisplay)
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('sensor_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Time-based features
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek

# Lag features
for col in ['temperature', 'vibration', 'pressure']:
    for lag in range(1, 4):
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

# Rolling features
for col in ['temperature', 'vibration', 'pressure']:
    df[f'{col}_roll_mean3'] = df[col].rolling(window=3).mean()
    df[f'{col}_roll_std3'] = df[col].rolling(window=3).std()

# Drop NaN values
df.dropna(inplace=True)

# Define features and target
X = df.drop(columns=['timestamp', 'failure'])
y = df['failure']

# Time-based train-test split 
train_size = int(0.8 * len(df))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Initialize model with class_weight='balanced'
model = RandomForestClassifier(
    class_weight={0:1, 1:20},  # 20x penalty for missing failures
    n_estimators=200,
    max_depth=10,              # Deeper trees to capture rare patterns
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for positive class

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Precision, Recall, F1
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_probs):.2f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Precision-Recall Curve
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_probs)
plt.plot(recall_vals, precision_vals)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Display thresholds for interpretation
threshold_df = pd.DataFrame({
    'Threshold': pr_thresholds,
    'Precision': precision_vals[:-1],
    'Recall': recall_vals[:-1]
})
print(threshold_df.head())
