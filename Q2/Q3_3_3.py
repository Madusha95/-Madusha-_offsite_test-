import pandas as pd
import string
import spacy
from textblob import Word
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

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
num_classes = len(label_encoder.classes_)

# Train-Test Split
train_df, test_df = train_test_split(
    df[['cleaned_text', 'intent_encoded']],
    test_size=0.2,
    random_state=42,
    stratify=df['intent_encoded']
)

# Tokenize text for LSTM
# Convert text to sequences of integers
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_df['cleaned_text'])

# Convert text to sequences of integers
X_train_seq = tokenizer.texts_to_sequences(train_df['cleaned_text'])
X_test_seq = tokenizer.texts_to_sequences(test_df['cleaned_text'])

# Pad sequences to ensure uniform length
max_sequence_length = 50  
X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length)

# Convert labels to categorical format for neural network
y_train_cat = to_categorical(train_df['intent_encoded'], num_classes=num_classes)
y_test_cat = to_categorical(test_df['intent_encoded'], num_classes=num_classes)

# Build LSTM model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100  # Dimension for word embeddings

print("\nBuilding LSTM model...")
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(SpatialDropout1D(0.2))  # Helps prevent overfitting
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))  # LSTM layer
model.add(Dense(64, activation='relu'))  # Dense hidden layer
model.add(Dropout(0.3))  # Additional dropout for regularization
model.add(Dense(num_classes, activation='softmax'))  # Output layer

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
print(model.summary())

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
print("\nTraining LSTM model...")
history = model.fit(
    X_train_padded, y_train_cat,
    validation_data=(X_test_padded, y_test_cat),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop]
)

# Evaluate model
train_loss, train_acc = model.evaluate(X_train_padded, y_train_cat, verbose=0)
test_loss, test_acc = model.evaluate(X_test_padded, y_test_cat, verbose=0)

print("\nEvaluation results:")
print(f"Training Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")
