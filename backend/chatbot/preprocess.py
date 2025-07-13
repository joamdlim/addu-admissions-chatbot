import nltk
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_text(text):
    _ = PunktSentenceTokenizer()
    # Lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove punctuation and stopwords
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]

    # Stemming
    stemmed = [stemmer.stem(t) for t in tokens]

    return stemmed
