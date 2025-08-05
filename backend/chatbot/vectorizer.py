import os
import json
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
from preprocess import preprocess_text

# Check if Word2Vec model exists, otherwise use a simple fallback
WORD2VEC_PATH = "chatbot/embeddings/GoogleNews-vectors-negative300.bin"
word2vec_model = None

try:
    if os.path.exists(WORD2VEC_PATH):
        word2vec_model = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
        print("✅ Loaded Word2Vec model")
    else:
        print("⚠️ Word2Vec model not found. Using TF-IDF only.")
except Exception as e:
    print(f"⚠️ Error loading Word2Vec model: {e}")

def load_json_docs(directory: str) -> List[Dict[str, str]]:
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                data = json.load(file)
                documents.append(data)
    return documents

def build_tfidf_vectorizer(corpus: list[str]) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    return vectorizer

def compute_word2vec_vector(tokens: list[str]) -> np.ndarray:
    if word2vec_model is None:
        return np.zeros(300)  # Return zeros if model not available
    
    vectors = [word2vec_model[word] for word in tokens if word in word2vec_model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)

def vectorize_documents(json_docs: list[dict]):
    # Check if documents list is empty
    if not json_docs:
        print("⚠️ No documents found!")
        return [], []
    
    # Check if documents have the expected structure
    if "content" not in json_docs[0]:
        print("⚠️ Documents don't have 'content' field!")
        print(f"Available fields: {list(json_docs[0].keys())}")
        # Try to use a different field or create empty content
        for doc in json_docs:
            doc["content"] = doc.get("text", "") or doc.get("body", "") or ""
    
    corpus = [" ".join(preprocess_text(doc["content"])) for doc in json_docs]

    tfidf = build_tfidf_vectorizer(corpus)
    tfidf_vectors = tfidf.transform(corpus).toarray()

    # Only use Word2Vec if available
    if word2vec_model:
        word2vec_vectors = [compute_word2vec_vector(preprocess_text(doc["content"])) for doc in json_docs]
        # Combine (e.g. concatenate)
        hybrid_vectors = [np.concatenate((tfidf_vec, w2v_vec)) for tfidf_vec, w2v_vec in zip(tfidf_vectors, word2vec_vectors)]
        return hybrid_vectors, json_docs
    else:
        # Just use TF-IDF vectors if Word2Vec not available
        return tfidf_vectors, json_docs