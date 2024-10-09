
# Download Sentiment140 Dataset using Kaggle
!mkdir -p ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d kazanova/sentiment140
!unzip sentiment140.zip

# Load Dataset into Pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'training.1600000.processed.noemoticon.csv'
columns = ['target', 'id', 'date', 'flag', 'user', 'text']
df = pd.read_csv(file_path, encoding='ISO-8859-1', header=None, names=columns)

# Basic Dataset Info
print(f"Dataset shape: {df.shape}")
df.info()

# Check for duplicate and missing values
print(f"Number of duplicate rows: {df.duplicated().sum()}")
print("Missing values in each column:")
print(df.isnull().sum())

# Sentiment Mapping and Visualization
df['sentiment'] = df['target'].map({0: 'Negative', 4: 'Positive'})

def plot_sentiment_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='sentiment', data=df)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

plot_sentiment_distribution(df)

# Text Length Analysis
df['text_length'] = df['text'].apply(len)

def plot_text_length_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['text_length'], bins=50, kde=True)
    plt.title('Distribution of Tweet Lengths')
    plt.xlabel('Tweet Length')
    plt.ylabel('Frequency')
    plt.show()

plot_text_length_distribution(df)

def plot_tweet_length_by_sentiment(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='sentiment', y='text_length', data=df)
    plt.title('Tweet Lengths by Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Tweet Length')
    plt.show()

plot_tweet_length_by_sentiment(df)

# Convert 'date' column to datetime and handle invalid dates
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date'], inplace=True)
df.set_index('date', inplace=True)

def plot_daily_sentiment_trends(df):
    daily_sentiment = df.groupby([pd.Grouper(freq='D'), 'sentiment'])['sentiment'].count().unstack()
    plt.figure(figsize=(15, 7))
    daily_sentiment.plot(ax=plt.gca())
    plt.title('Daily Sentiment Trends')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend(['Negative', 'Positive'])
    plt.show()

plot_daily_sentiment_trends(df)

# Function to Extract and Visualize N-Grams
from sklearn.feature_extraction.text import CountVectorizer

def plot_ngrams(texts, ngram_range=(1, 2), top_n=10):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    ngrams = vectorizer.fit_transform(texts)
    ngram_counts = ngrams.sum(axis=0).A1
    ngram_features = vectorizer.get_feature_names_out()
    ngram_df = pd.DataFrame({'ngram': ngram_features, 'count': ngram_counts})
    top_ngrams = ngram_df.sort_values('count', ascending=False).head(top_n)

    plt.figure(figsize=(10, 5))
    sns.barplot(x='count', y='ngram', data=top_ngrams)
    plt.title(f'Top {top_n} N-grams')
    plt.xlabel('Count')
    plt.ylabel('N-grams')
    plt.show()

# Plotting Bi-grams and Tri-grams
plot_ngrams(df['text'], ngram_range=(2, 2), top_n=10)
plot_ngrams(df['text'], ngram_range=(3, 3), top_n=10)

# Generate Word Clouds
from wordcloud import WordCloud

def generate_wordcloud(text, background_color='white', title='Word Cloud'):
    wordcloud = WordCloud(width=800, height=400, background_color=background_color).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

positive_text = ' '.join(df[df['sentiment'] == 'Positive']['text'])
negative_text = ' '.join(df[df['sentiment'] == 'Negative']['text'])

generate_wordcloud(positive_text, background_color='white', title='Word Cloud for Positive Sentiment')
generate_wordcloud(negative_text, background_color='black', title='Word Cloud for Negative Sentiment')

# Preparing Data for Model Training
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import numpy as np

data_pos = df[df['target'] == 4].sample(100000)
data_neg = df[df['target'] == 0].sample(100000)
dataset = pd.concat([data_pos, data_neg]).sample(frac=1).reset_index(drop=True)

X = list(dataset['text'])
y = list(dataset['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(X_train, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)

# TensorFlow Dataset Preparation
import tensorflow as tf

train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test))

# Model Training
from transformers import TFDistilBertForSequenceClassification, TrainingArguments, Trainer

model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.1,
    logging_dir='./logs',
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Model Evaluation and Saving
from sklearn.metrics import classification_report
y_pred = np.argmax(trainer.predict(test_dataset).predictions, axis=1)
print(classification_report(y_test, y_pred))

# Save Model to Google Drive
from google.colab import drive
drive.mount('/content/drive')

model.save_pretrained("/content/drive/MyDrive/sentiment_analysis_model")
tokenizer.save_pretrained("/content/drive/MyDrive/sentiment_analysis_model")

# Streamlit App for User Interaction
import streamlit as st

st.title('Text Classification with DistilBERT')

def load_model_and_tokenizer():
    model = TFDistilBertForSequenceClassification.from_pretrained("/content/drive/MyDrive/sentiment_analysis_model")
    tokenizer = DistilBertTokenizerFast.from_pretrained("/content/drive/MyDrive/sentiment_analysis_model")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

user_input = st.text_area("Enter the text to classify")

if user_input:
    inputs = tokenizer(user_input, return_tensors="tf", truncation=True, padding=True)
    outputs = model(inputs)
    predictions = tf.argmax(outputs.logits, axis=-1)
    st.write(f"Predicted class: {predictions.numpy()[0]}")
