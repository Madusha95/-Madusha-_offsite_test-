import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import string
import spacy
from textblob import Word
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Load the data
df = pd.read_csv('intent_classification_data.csv')

# Display original examples
print("Original Examples:")
print(df.head(5))

# Convert to Lowercase
df['text_clean'] = df['text'].str.lower()
print("\nAfter lowercase conversion:")
print(df[['text', 'text_clean']].head(3))

# Remove Punctuation and Special Characters
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

df['text_clean'] = df['text_clean'].apply(remove_punctuation)
print("\nAfter punctuation removal:")
print(df[['text', 'text_clean']].head(3))

# Tokenization
df['tokens'] = df['text_clean'].str.split()
print("\nAfter tokenization:")
print(df[['text_clean', 'tokens']].head(3))

# Remove Stopwords
nlp = spacy.blank("en")

def remove_stopwords_spacy(token_list):
    return [word for word in token_list if not nlp.vocab[word].is_stop]

df['tokens'] = df['tokens'].apply(remove_stopwords_spacy)
print("\nAfter stopword removal:")
print(df[['text_clean', 'tokens']].head(3))

# Lemmatization
def textblob_lemmatize(token_list):
    return [Word(word).lemmatize() for word in token_list]

df['tokens'] = df['tokens'].apply(textblob_lemmatize)
print("\nAfter lemmatization:")
print(df[['text_clean', 'tokens']].head(3))

# Create cleaned text column
df['cleaned_text'] = df['tokens'].apply(' '.join)
print("\nOriginal vs Cleaned Text:")
print(df[['text', 'cleaned_text']].head(5))

# BERT Embeddings
print("\nGenerating BERT embeddings...")

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    # Tokenize the text and convert to tensor
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use mean of last hidden state as sentence embedding
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Apply BERT embedding to cleaned text
df['bert_embedding'] = df['cleaned_text'].apply(get_bert_embedding)

# Convert BERT embeddings to numpy array for easier use in models
bert_embeddings = np.array(df['bert_embedding'].tolist())
print(f"\nBERT embeddings shape: {bert_embeddings.shape}")

# Print final dataframe with BERT embeddings
print("\nFinal dataframe with BERT embeddings:")
print(df.head(3))