from flask import Flask, render_template, request
import pickle
import re
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

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

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the text input from the form
        text = request.form['text']

        # Preprocess the text
        processed_text = preprocess_text(text)

        # Predict the sentiment using the loaded SVM model
        prediction = model.predict([processed_text])[0]

        # Convert prediction to text label
        sentiment = "Positive" if prediction == 1 else "Negative"

        return render_template('index.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
