import pandas as pd
import string
import spacy
from textblob import Word
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import numpy as np

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

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_texts(texts):
    return tokenizer(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

train_encodings = tokenize_texts(train_df['cleaned_text'])
test_encodings = tokenize_texts(test_df['cleaned_text'])

# Dataset class
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IntentDataset(train_encodings, train_df['intent_encoded'].values)
test_dataset = IntentDataset(test_encodings, test_df['intent_encoded'].values)

# Accuracy metric function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

# Label mappings
id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
label2id = {label: i for i, label in id2label.items()}

# Load pre-trained model with new classification head
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_encoder.classes_),
    id2label=id2label,
    label2id=label2id
)

# Training arguments (compatible version)
training_args = TrainingArguments(
    output_dir='./fine_tune_results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    do_eval=True,  
    warmup_steps=100,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=1
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train model
print("\nStarting BERT fine-tuning...")
trainer.train()

# Evaluate model
results = trainer.evaluate()
print("\nFine-tuning evaluation results:")
print(f"Accuracy: {results['eval_accuracy']:.2%}")
print(f"Loss: {results['eval_loss']:.4f}")

# Save final model & tokenizer
model.save_pretrained('./fine_tuned_intent_classifier')
tokenizer.save_pretrained('./fine_tuned_intent_classifier')
print("\nSaved fine-tuned model to './fine_tuned_intent_classifier'")
