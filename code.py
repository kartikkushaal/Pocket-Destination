!pip install pandas numpy textblob nltk gensimimport pandas as pd
import numpy as np
import re
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim.downloader as api
import shutil
import os
data_dir = os.path.join(os.path.expanduser("~"), ".cache", "gensim-data")
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
w2v_model = api.load('word2vec-google-news-300')

# Function to preprocess the reviews
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_text = [word for word in word_tokens if word.isalpha() and word not in stop_words]
    return ' '.join(filtered_text)

# Function to compute sentiment using embeddings and TextBlob
def show_opinion_with_embeddings(x):
    if not x or pd.isnull(x):
        return 0, "Neutral"

    stop_words = set(stopwords.words())
    tokenized_text = word_tokenize(x)
    filtered_words = [word for word in tokenized_text if word.lower() not in stop_words]

    if not filtered_words:
        return 0, "Neutral"

    word_embeddings = [w2v_model[word] for word in filtered_words if word in w2v_model]
    if not word_embeddings:
        return 0, "Neutral"
    !pip install pandas numpy textblob nltk gensim
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim.downloader as api
import shutil
import os

data_dir = os.path.join(os.path.expanduser("~"), ".cache", "gensim-data")
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab') # Download the punkt_tab data package

w2v_model = api.load('word2vec-google-news-300')

# Function to preprocess the reviews
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_text = [word for word in word_tokens if word.isalpha() and word not in stop_words]
    return ' '.join(filtered_text)

# Function to compute sentiment using embeddings and TextBlob
def show_opinion_with_embeddings(x):
    if not x or pd.isnull(x):
        return 0, "Neutral"

    stop_words = set(stopwords.words())
    tokenized_text = word_tokenize(x)
    filtered_words = [word for word in tokenized_text if word.lower() not in stop_words]

    if not filtered_words:
        return 0, "Neutral"

    word_embeddings = [w2v_model[word] for word in filtered_words if word in w2v_model]
    if not word_embeddings:
        return 0, "Neutral"

    # Calculate sentiment using TextBlob
    sentiment_polarity = TextBlob(x).sentiment.polarity
    return round(sentiment_polarity, 2), "Positive" if sentiment_polarity > 0.01 else "Negative" if sentiment_polarity < -0.01 else "Neutral"

# Function to combine sentiment from text and numeric rating
def combined_sentiment(text, rating):
    text_sentiment, text_opinion = show_opinion_with_embeddings(text)
    combined_sentiment_score = text_sentiment * rating if pd.notnull(rating) else text_sentiment

    # Determine overall opinion based on combined sentiment
    if combined_sentiment_score > 0.01:
        overall_opinion = "Positive"
    elif combined_sentiment_score < -0.01:
        overall_opinion = "Negative"
    else:
        overall_opinion = "Neutral"

    return round(combined_sentiment_score, 2), overall_opinion
 # Preprocess review text
df['Hotel_Review'] = df['Hotel_Review'].apply(preprocess_text)
df['Food_Review'] = df['Food_Review'].apply(preprocess_text)
df['Tourist_Spot_Review'] = df['Tourist_Spot_Review'].apply(preprocess_text)

# Calculate sentiment scores and add them as new columns
df['Hotel_Sentiment'], _ = zip(*df['Hotel_Review'].apply(show_opinion_with_embeddings)) # Apply show_opinion_with_embeddings to the review columns and store the sentiment scores in new columns
df['Food_Sentiment'], _ = zip(*df['Food_Review'].apply(show_opinion_with_embeddings))
df['Tourist_Spot_Sentiment'], _ = zip(*df['Tourist_Spot_Review'].apply(show_opinion_with_embeddings))

def combined_sentiment(text, rating):
    text_sentiment, text_opinion = show_opinion_with_embeddings(text)
    combined_sentiment_score = text_sentiment * rating if pd.notnull(rating) else text_sentiment  # Handles np.nan gracefully

# Function to get recommendations for a specific city
def recommend_best(state, city):
    city_data = df[(df['State'].str.contains(state, case=False)) & (df['City'].str.contains(city, case=False))]

    if city_data.empty:
        print(f"No data found for {city}, {state}.")
        return

    best_hotel = city_data.sort_values(by='Hotel_Sentiment', ascending=False).iloc[0]
    best_food = city_data.sort_values(by='Food_Sentiment', ascending=False).iloc[0]
    best_tourist_spot = city_data.sort_values(by='Tourist_Spot_Sentiment', ascending=False).iloc[0]

    print(f"Best Hotel: {best_hotel['Hotel']} - Sentiment: {best_hotel['Hotel_Sentiment']}")
    print(f"Best Food: {best_food['Food']} - Sentiment: {best_food['Food_Sentiment']}")
    print(f"Best Tourist Spot: {best_tourist_spot['Tourist_Spot']} - Sentiment: {best_tourist_spot['Tourist_Spot_Sentiment']}")

# Input from user
state_input = input("Enter the state: ")
city_input = input("Enter the city: ")

# Get recommendations for the user-specified city
recommend_best(state_input, city_input)
