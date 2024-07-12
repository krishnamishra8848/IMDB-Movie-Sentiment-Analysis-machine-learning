import streamlit as st
import pickle
import re
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the SVM model
with open('svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Tokenization and stemming function
stemmer = PorterStemmer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove punctuation and stop words, and stem the words
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(tokens)

# Define the Streamlit app
def main():
    st.title('Movie Sentiment Analysis')

    # Text input for sentiment analysis
    text = st.text_area('Enter text for sentiment analysis:', '')

    if st.button('Predict Sentiment'):
        if text:
            # Preprocess the text
            processed_text = preprocess_text(text)

            # Predict the sentiment using the loaded SVM model
            prediction = model.predict([processed_text])[0]

            # Convert prediction to text label
            sentiment = "Positive" if prediction == 1 else "Negative"

           

            st.header('Sentiment Prediction:')
            if sentiment == "Positive":
                st.success(sentiment)
            else:
                st.error(sentiment)

# Run the Streamlit app
if __name__ == '__main__':
    main()
