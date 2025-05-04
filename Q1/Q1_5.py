import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load the data
df = pd.read_csv('sensor_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Engineer time-based features (as we did before)
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek

# Lag features
for col in ['temperature', 'vibration', 'pressure']:
    for lag in range(1, 4):
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

# Rolling stats
for col in ['temperature', 'vibration', 'pressure']:
    df[f'{col}_roll_mean3'] = df[col].rolling(window=3).mean()
    df[f'{col}_roll_std3'] = df[col].rolling(window=3).std()

# Drop rows with NaN values (from lags/rolling)
df.dropna(inplace=True)

# Define X (features) and y (target)
X = df.drop(columns=['timestamp', 'failure'])
y = df['failure']

print(X.head(10))  # Display the first few rows of the DataFrame to verify the new features
print(y.head(10))  # Display the first few rows of the target variable


train_size = int(0.8 * len(df))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

#Evaluate Model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#Save the Trained Model
joblib.dump(model, 'rf_machine_failure_model.pkl')
print("Model saved successfully!")


#1.5
from sklearn.metrics import accuracy_score

#Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#Precision
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
print("Precision:", precision)

#Recall
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)
print("Recall:", recall)

#F1 Score
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)
print("F1-Score:", f1)

#Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

#ROC Curve & AUC (Area Under Curve)
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_probs = model.predict_proba(X_test)[:, 1]  # get probability for class 1
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc_score(y_test, y_probs)))
plt.plot([0,1], [0,1], 'k--')  # random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curve')
plt.show()

#Precision-Recall Curve

from sklearn.metrics import precision_recall_curve

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_probs)

plt.plot(recall_vals, precision_vals)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
