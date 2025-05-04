import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load the data
df = pd.read_csv('intent_classification_data.csv')

# Display original examples
print("Original Examples:")
print(df.head(5))


#Convert to Lowercase
df['text_clean'] = df['text'].str.lower()
print("\nAfter lowercase conversion:")
print(df[['text', 'text_clean']].head(3))

#Remove Punctuation and Special Characters
import string

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

df['text_clean'] = df['text_clean'].apply(remove_punctuation)
print("\nAfter punctuation removal:")
print(df[['text', 'text_clean']].head(3))

#Tokenization
df['tokens'] = df['text_clean'].str.split()
print(df[['text_clean', 'tokens']].head(3))

#Remove Stopwords
import spacy
nlp = spacy.blank("en")  # Use a blank English model to avoid requiring a download

def remove_stopwords_spacy(token_list):
    return [word for word in token_list if not nlp.vocab[word].is_stop]

df['tokens'] = df['tokens'].apply(remove_stopwords_spacy)
print("\nAfter stopword removal:")
print(df[['text_clean', 'tokens']].head(3))

# Lemmatization
from textblob import Word  # Install with: pip install textblob

def textblob_lemmatize(token_list):
    return [Word(word).lemmatize() for word in token_list]

df['tokens'] = df['tokens'].apply(textblob_lemmatize)
print("\nAfter lemmatization:")
print(df[['text_clean', 'tokens']].head(3))

#Cleaned Text
df['cleaned_text'] = df['tokens'].apply(' '.join)
print("\nOriginal vs Cleaned Text:")
print(df[['text', 'cleaned_text']].head(5))

#Visualizations
#Token Frequency Distribution
# Flatten all tokens
all_tokens = [token for sublist in df['tokens'] for token in sublist]

# Get top 20 most common tokens
token_counts = Counter(all_tokens)
top_tokens = token_counts.most_common(20)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=[count for token, count in top_tokens], 
            y=[token for token, count in top_tokens])
plt.title('Top 20 Most Frequent Tokens')
plt.xlabel('Count')
plt.ylabel('Token')
plt.show()

#Token Count by Intent
# Create a DataFrame for tokens per intent
intent_tokens = df.explode('tokens').groupby(['intent', 'tokens']).size().reset_index(name='count')

# Get top tokens per intent
top_intent_tokens = intent_tokens.sort_values(['intent', 'count'], ascending=False).groupby('intent').head(3)

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(data=top_intent_tokens, x='count', y='intent', hue='tokens')
plt.title('Top Tokens by Intent')
plt.xlabel('Count')
plt.ylabel('Intent')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


