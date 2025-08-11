import os
import io
import json
import numpy as np
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
import chromadb # Import chromadb to access client methods like delete_collection
import re # Import the re module for regular expressions

from chatbot.supabase_client import get_supabase_client
from chatbot.chroma_connection import ChromaService
from chatbot.preprocess import preprocess_text # Ensure this is imported for preprocessing

# --- Configuration for Embeddings (Matches FastHybridChatbot) ---
# Assuming these paths are relative to the 'backend' directory
EMBEDDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings")
WORD2VEC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "GoogleNews-vectors-negative300.bin")
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "metadata.json")

import joblib  # add this import

TFIDF_VECTORIZER_PATH = os.path.join(EMBEDDINGS_DIR, "tfidf_vectorizer.pkl")
W2V_DIM = 300

# --- Global/Singleton for Vectorizer and Word2Vec Model ---
# This will be initialized once
tfidf_vectorizer = None
word2vec_model = None

documents = [] # To hold the corpus for TF-IDF

def initialize_embedding_models():
    """
    Initializes the TF-IDF vectorizer and Word2Vec model.
    This should be called once at the start of the application or script.
    """
    global tfidf_vectorizer, word2vec_model

    # Load or fit TF-IDF vectorizer once, then persist
    if os.path.exists(TFIDF_VECTORIZER_PATH):
        tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        print("✅ Loaded persisted TF-IDF vectorizer.")
    else:
        # Build vocabulary from current metadata.json ONCE
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"metadata.json not found at {METADATA_PATH} to build TF-IDF.")
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        corpus = [" ".join(preprocess_text(doc["content"])) for doc in docs]
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2), lowercase=False)
        tfidf_vectorizer.fit(corpus)
        joblib.dump(tfidf_vectorizer, TFIDF_VECTORIZER_PATH)
        print(f"✅ Fitted and saved TF-IDF vectorizer with {len(tfidf_vectorizer.get_feature_names_out())} features.")

    # Load Word2Vec (300-dim)
    if os.path.exists(WORD2VEC_PATH):
        word2vec_model = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
        print("✅ Loaded Word2Vec (300-dim).")
    else:
        raise FileNotFoundError(f"Required Word2Vec model not found at {WORD2VEC_PATH}.")


def extract_text_from_pdf(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def embed_text(text):
    """
    Generates a hybrid embedding (TF-IDF + Word2Vec) for the given text
    using the globally initialized models.
    """
    if tfidf_vectorizer is None or word2vec_model is None:
        raise ValueError("Embedding models not initialized. Call initialize_embedding_models() first.")

    tokens = preprocess_text(text)
    processed_text = " ".join(tokens)

    tfidf_vec = tfidf_vectorizer.transform([processed_text]).toarray()[0]  # fixed length: TFIDF_VOCAB_SIZE
    vectors = [word2vec_model[w] for w in tokens if w in word2vec_model]
    w2v_vec = np.mean(vectors, axis=0) if vectors else np.zeros(W2V_DIM)

    hybrid = np.concatenate((tfidf_vec, w2v_vec))  # fixed length: TFIDF_VOCAB_SIZE + 300
    return hybrid.tolist()

def process_and_store_pdf_in_chroma(pdf_bytes: bytes, pdf_file_name: str):
    """
    Processes a PDF from bytes, extracts text, generates embeddings,
    and stores it in a ChromaDB collection.
    """
    initialize_embedding_models()

    chroma_client = ChromaService.get_client()
    collection_name = os.getenv("CHROMA_COLLECTION", "documents")
    collection = chroma_client.get_or_create_collection(name=collection_name)
    print(f"✅ Using collection '{collection_name}'")

    text = extract_text_from_pdf(pdf_bytes)
    embedding = embed_text(text)

    doc_id = os.path.splitext(pdf_file_name)[0]
    collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"filename": pdf_file_name, "source": "pdf_scrape"}]
    )

def main():
    # initialize_embedding_models() # No longer needed here as it's called in the new function

    supabase = get_supabase_client()

    # List all PDFs in the original-pdfs bucket
    pdf_files_in_bucket = supabase.storage.from_("original-pdfs").list()
    
    # Filter for actual PDF files and extract names
    pdf_names = [f['name'] for f in pdf_files_in_bucket if f['name'].endswith('.pdf')]

    if not pdf_names:
        print("No PDF files found in the bucket.")
        return

    # Process each PDF file
    for pdf_file in pdf_names:
        # Download the PDF
        pdf_bytes = supabase.storage.from_("original-pdfs").download(pdf_file)
        process_and_store_pdf_in_chroma(pdf_bytes, pdf_file)
        print("-" * 50) # Separator for clarity between PDF processing

if __name__ == "__main__":
    main()