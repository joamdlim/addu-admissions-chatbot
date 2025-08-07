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
    global tfidf_vectorizer, word2vec_model, documents

    # Load metadata (corpus for TF-IDF)
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        print(f"✅ Loaded metadata for {len(documents)} documents for TF-IDF corpus.")
    else:
        print(f"⚠️ Metadata file not found at: {METADATA_PATH}. TF-IDF will not be initialized.")
        # If metadata is not found, we cannot build TF-IDF. This is a critical warning.
        # Consider creating dummy documents or handling this case more robustly.
        return

    # Build TF-IDF vectorizer
    if documents:
        corpus = [" ".join(preprocess_text(doc["content"])) for doc in documents]
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(corpus)
        print("✅ Initialized global TF-IDF vectorizer.")
    else:
        print("⚠️ No documents in metadata to build TF-IDF vectorizer.")

    # Load Word2Vec model
    if os.path.exists(WORD2VEC_PATH):
        try:
            word2vec_model = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
            print("✅ Loaded global Word2Vec model.")
        except Exception as e:
            print(f"⚠️ Failed to load Word2Vec model from {WORD2VEC_PATH}: {e}. Will proceed without Word2Vec.")
    else:
        print(f"⚠️ Word2Vec model not found at: {WORD2VEC_PATH}. Will proceed without Word2Vec.")


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
    if tfidf_vectorizer is None and word2vec_model is None:
        raise ValueError("Embedding models are not initialized. Call initialize_embedding_models() first.")

    tokens = preprocess_text(text)
    processed_text = " ".join(tokens)

    # TF-IDF vector
    if tfidf_vectorizer:
        tfidf_vector = tfidf_vectorizer.transform([processed_text]).toarray()[0]
    else:
        # Fallback if TF-IDF wasn't initialized, create a zero vector of appropriate size
        # This size should ideally come from a fixed vocabulary if no model is loaded.
        # For this test, if TF-IDF model is missing, its contribution to the final vector will be zero.
        # If your metadata.json is empty or not found, tfidf_vectorizer will be None.
        # For a consistent dimension, this might need a default size.
        # However, the primary fix is to ensure metadata.json is present and loaded.
        tfidf_vector = np.zeros(0)

    # Word2Vec vector
    if word2vec_model:
        vectors = [word2vec_model[word] for word in tokens if word in word2vec_model]
        w2v_vector = np.mean(vectors, axis=0) if vectors else np.zeros(300) # Assuming 300 dim for GoogleNews
    else:
        # Fallback if Word2Vec wasn't initialized
        w2v_vector = np.zeros(0)

    # Concatenate only if both are present, otherwise return the one that is.
    if tfidf_vector.size > 0 and w2v_vector.size > 0:
        hybrid_vector = np.concatenate((tfidf_vector, w2v_vector))
    elif tfidf_vector.size > 0:
        hybrid_vector = tfidf_vector
    elif w2v_vector.size > 0:
        hybrid_vector = w2v_vector
    else:
        raise RuntimeError("No embedding models available to create a vector. Check metadata.json and Word2Vec path.")

    return hybrid_vector.tolist()

def process_and_store_pdf_in_chroma(pdf_bytes: bytes, pdf_file_name: str):
    """
    Processes a PDF from bytes, extracts text, generates embeddings,
    and stores it in a ChromaDB collection.
    """
    initialize_embedding_models() # Ensure models are initialized

    # Define the collection name based on the PDF file name
    base_name = os.path.splitext(pdf_file_name)[0]
    pdf_collection_name = re.sub(r'[^a-zA-Z0-9._-]', '_', base_name).lower()
    pdf_collection_name = pdf_collection_name.strip('_')
    if len(pdf_collection_name) < 3:
        pdf_collection_name = "pdf_" + pdf_collection_name

    chroma_client = ChromaService.get_client()
    pdf_chroma_collection = chroma_client.get_or_create_collection(name=pdf_collection_name)
    print(f"✅ Ensured collection '{pdf_collection_name}' exists.")

    print(f"Processing: {pdf_file_name}")

    text = extract_text_from_pdf(pdf_bytes)
    print("Extracted text preview:\n", text[:500], "\n---\n")

    embedding = embed_text(text)
    print(f"Generated embedding with dimension: {len(embedding)}")

    doc_id = os.path.splitext(pdf_file_name)[0]
    pdf_chroma_collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"filename": pdf_file_name, "source": "pdf_scrape"}]
    )
    print(f"✅ Stored embedding for {pdf_file_name} in '{pdf_collection_name}' collection.")

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