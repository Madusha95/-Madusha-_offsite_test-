import pandas as pd
import string
import spacy
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('intent_classification_data.csv')
print("Original Examples:")
print(df.head(5))

# Text Preprocessing
df['text_clean'] = df['text'].str.lower()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

df['text_clean'] = df['text_clean'].apply(remove_punctuation)
df['tokens'] = df['text_clean'].str.split()

# Remove stopwords using spaCy
nlp = spacy.blank("en")
def remove_stopwords(token_list):
    return [word for word in token_list if not nlp.vocab[word].is_stop]

df['tokens'] = df['tokens'].apply(remove_stopwords)

# Lemmatization using TextBlob
def lemmatize_tokens(token_list):
    return [Word(word).lemmatize() for word in token_list]

df['tokens'] = df['tokens'].apply(lemmatize_tokens)

# Final cleaned text
df['cleaned_text'] = df['tokens'].apply(' '.join)

print("\nOriginal vs Cleaned Text:")
print(df[['text', 'cleaned_text']].head(5))

# Encode labels
label_encoder = LabelEncoder()
df['intent_encoded'] = label_encoder.fit_transform(df['intent'])

# Train-Test Split
train_df, test_df = train_test_split(
    df[['cleaned_text', 'intent_encoded']],
    test_size=0.2,
    random_state=42,
    stratify=df['intent_encoded']
)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,      # Increase features
    ngram_range=(1, 2),     # Use unigrams and bigrams
    stop_words='english',    # Built-in stopwords
    min_df=2,               # Ignore very rare words
    max_df=0.95             # Ignore very common words
)
X_train = vectorizer.fit_transform(train_df['cleaned_text'])
X_test = vectorizer.transform(test_df['cleaned_text'])
y_train = train_df['intent_encoded']
y_test = test_df['intent_encoded']

# Train Logistic Regression model
print("\nTraining Logistic Regression model...")
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

# Generate predictions
train_preds = logreg.predict(X_train)
test_preds = logreg.predict(X_test)

# Get predicted probabilities for the test set
test_probs = logreg.predict_proba(X_test)

# Label mappings for reference
id2label = {i: label for i, label in enumerate(label_encoder.classes_)}

# Evaluation metrics
def evaluate_model(y_true, y_pred, label_names):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names))
    
    print("\nOverall Metrics:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-score: {f1:.2%}")
    
    return accuracy, precision, recall, f1

# Evaluate on test set
print("\nTest Set Evaluation:")
label_names = [id2label[i] for i in range(len(id2label))]
test_metrics = evaluate_model(y_test, test_preds, label_names)

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, label_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

print("\nConfusion Matrix Visualization:")
plot_confusion_matrix(y_test, test_preds, label_names)

# Display some example predictions
test_df['predicted_intent'] = [id2label[p] for p in test_preds]
test_df['actual_intent'] = [id2label[a] for a in y_test]

print("\nSample Predictions:")
print(test_df[['cleaned_text', 'actual_intent', 'predicted_intent']].head(10))

# Display model coefficients for top features (for interpretation)
def show_top_features(vectorizer, model, class_idx, n=10):
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[class_idx]
    top_features = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)[:n]
    bottom_features = sorted(zip(feature_names, coefs), key=lambda x: x[1])[:n]
    
    print(f"\nTop features for class '{id2label[class_idx]}':")
    for feat, coef in top_features:
        print(f"{feat}: {coef:.3f}")
    
    print(f"\nBottom features for class '{id2label[class_idx]}':")
    for feat, coef in bottom_features:
        print(f"{feat}: {coef:.3f}")

# Show features for a few classes
for class_idx in range(min(3, len(id2label))):  # Show first 3 classes
    show_top_features(vectorizer, logreg, class_idx)