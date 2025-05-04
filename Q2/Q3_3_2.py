import pandas as pd
import string
import spacy
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
# Try different vectorization approaches
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

# Evaluate model
train_preds = logreg.predict(X_train)
test_preds = logreg.predict(X_test)

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

print("\nEvaluation results:")
print(f"Training Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")

# Label mappings for reference
id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
print("\nLabel mappings:")
for id, label in id2label.items():
    print(f"{id}: {label}")