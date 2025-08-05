import os
import io
import json
import numpy as np
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
import chromadb # Import chromadb to access client methods like delete_collection

from supabase_client import get_supabase_client
from chroma_connection import ChromaService
from preprocess import preprocess_text # Ensure this is imported for preprocessing

# --- Configuration for Embeddings (Matches FastHybridChatbot) ---
# Assuming these paths are relative to the 'backend' directory
EMBEDDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings")
WORD2VEC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "GoogleNews-vectors-negative300.bin")
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "metadata.json")

# --- New: Define the name for your PDF collection ---
PDF_CHROMA_COLLECTION_NAME = "pdf_scraped_documents"

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

def main():
    # Initialize embedding models once
    initialize_embedding_models()

    # Get ChromaDB client instance
    chroma_client = ChromaService.get_client()

    # --- IMPORTANT for testing: Delete the new PDF collection if it exists ---
    # This ensures a clean slate for the PDF collection's dimension each test run.
    try:
        chroma_client.delete_collection(name=PDF_CHROMA_COLLECTION_NAME)
        print(f"✅ Deleted existing '{PDF_CHROMA_COLLECTION_NAME}' collection to reset dimension.")
    except Exception as e:
        print(f"⚠️ Could not delete '{PDF_CHROMA_COLLECTION_NAME}' collection (might not exist yet): {e}")

    # Get or create the NEW PDF collection
    pdf_chroma_collection = chroma_client.get_or_create_collection(name=PDF_CHROMA_COLLECTION_NAME)
    print(f"✅ Ensured collection '{PDF_CHROMA_COLLECTION_NAME}' exists.")

    supabase = get_supabase_client()


    # List all PDFs in the original-pdfs bucket
    pdf_files = supabase.storage.from_("original-pdfs").list()
    pdf_file = next((f['name'] for f in pdf_files if f['name'].endswith('.pdf')), None)
    if not pdf_file:
        print("No PDF files found in the bucket.")
        return

    print(f"Processing: {pdf_file}")

    # Download the PDF
    pdf_bytes = supabase.storage.from_("original-pdfs").download(pdf_file)

    # Extract text
    text = extract_text_from_pdf(pdf_bytes)
    print("Extracted text preview:\n", text[:500], "\n---\n")

    # Embed the text using your actual embedding logic
    embedding = embed_text(text)
    print(f"Generated embedding with dimension: {len(embedding)}")


    # Store in the NEW ChromaDB collection
    doc_id = os.path.splitext(pdf_file)[0]
    pdf_chroma_collection.add( # Changed from chroma_collection to pdf_chroma_collection
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"filename": pdf_file, "source": "pdf_scrape"}]
    )
    print(f"✅ Stored embedding for {pdf_file} in '{PDF_CHROMA_COLLECTION_NAME}' collection.")

if __name__ == "__main__":
    main()