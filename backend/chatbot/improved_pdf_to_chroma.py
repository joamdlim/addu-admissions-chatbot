# backend/chatbot/improved_pdf_to_chroma.py

import os
import io
import re
import json
import numpy as np
from typing import List, Dict
import joblib

from chatbot.supabase_client import get_supabase_client
from chatbot.chroma_connection import ChromaService
from chatbot.preprocess import preprocess_text

# Import pdfplumber - now required
try:
    import pdfplumber
    print("✅ pdfplumber available")
except ImportError:
    raise ImportError("❌ pdfplumber is required. Please install it with: pip install pdfplumber")

# Configuration - matches existing setup
EMBEDDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings")
WORD2VEC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "GoogleNews-vectors-negative300.bin")
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "metadata.json")
TFIDF_VECTORIZER_PATH = os.path.join(EMBEDDINGS_DIR, "tfidf_vectorizer.pkl")
W2V_DIM = 300
PDF_BUCKET = "original-pdfs"

# Global models
tfidf_vectorizer = None
word2vec_model = None

def initialize_embedding_models():
    """Initialize TF-IDF vectorizer and Word2Vec model"""
    global tfidf_vectorizer, word2vec_model

    # Load or fit TF-IDF vectorizer
    if os.path.exists(TFIDF_VECTORIZER_PATH):
        tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        print("✅ Loaded persisted TF-IDF vectorizer.")
    else:
        # Build vocabulary from current metadata.json
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

    # Load Word2Vec
    if os.path.exists(WORD2VEC_PATH):
        from gensim.models import KeyedVectors
        word2vec_model = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
        print("✅ Loaded Word2Vec (300-dim).")
    else:
        raise FileNotFoundError(f"Required Word2Vec model not found at {WORD2VEC_PATH}.")

def extract_text_with_pdfplumber(pdf_bytes: bytes) -> str:
    """Extract text using pdfplumber with enhanced structure preservation"""
    text = ""
    pdf_file = io.BytesIO(pdf_bytes)
    
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    # Add page marker for better context
                    text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
                else:
                    print(f"⚠️ Warning: No text extracted from page {page_num + 1}")
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
            
        return text
        
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {str(e)}")

def clean_pdf_text(text: str) -> str:
    """Clean and normalize PDF text while preserving structure"""
    # Remove excessive whitespace but preserve paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n[ \t]+', '\n', text)  # Remove spaces after newlines
    text = re.sub(r'[ \t]+\n', '\n', text)  # Remove spaces before newlines
    
    # Fix minor formatting issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
    
    # Clean up step formatting - make it more consistent
    text = re.sub(r'STEP\s*(\d+)\s*:\s*', r'\n\nSTEP \1: ', text)
    
    # Clean up multiple newlines but preserve intentional breaks
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    
    return text.strip()

def analyze_step_order(text: str) -> Dict:
    """Analyze the order of steps in the text"""
    step_pattern = r'STEP\s+(\d+):'
    matches = re.findall(step_pattern, text, re.IGNORECASE)
    
    if not matches:
        return {"found": False, "steps": [], "order": "No steps found"}
    
    step_numbers = [int(match) for match in matches]
    expected_order = list(range(1, max(step_numbers) + 1))
    
    analysis = {
        "found": True,
        "steps": step_numbers,
        "expected": expected_order,
        "correct_order": step_numbers == expected_order,
        "order": "Correct" if step_numbers == expected_order else "Incorrect"
    }
    
    return analysis

def smart_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Chunk text intelligently by preserving logical sections"""
    chunks = []
    
    # Split by STEP sections first
    step_sections = re.split(r'(\n\n?STEP\s+\d+:)', text, flags=re.IGNORECASE)
    
    current_chunk = ""
    chunk_id = 0
    
    for i, section in enumerate(step_sections):
        if not section.strip():
            continue
        
        # Check if this is a STEP header
        if re.match(r'\n\n?STEP\s+\d+:', section, re.IGNORECASE):
            # This is a step header, combine with next section if available
            if i + 1 < len(step_sections):
                full_section = section + step_sections[i + 1]
                step_sections[i + 1] = ""  # Mark as processed
            else:
                full_section = section
        else:
            full_section = section
        
        if not full_section.strip():
            continue
            
        # If adding this section would exceed chunk size, finalize current chunk
        if len(current_chunk + "\n\n" + full_section) > chunk_size and current_chunk:
            chunks.append({
                'id': f'chunk_{chunk_id}',
                'content': current_chunk.strip(),
                'size': len(current_chunk),
                'type': 'section'
            })
            chunk_id += 1
            
            # Start new chunk with overlap
            if overlap > 0 and len(current_chunk) > overlap:
                overlap_text = current_chunk[-overlap:]
                # Try to start overlap at a good break point
                for break_char in ['\n\n', '\n', '. ']:
                    break_pos = overlap_text.find(break_char)
                    if break_pos > 50:
                        overlap_text = overlap_text[break_pos + len(break_char):]
                        break
                current_chunk = overlap_text + "\n\n" + full_section
            else:
                current_chunk = full_section
        else:
            if current_chunk:
                current_chunk += "\n\n" + full_section
            else:
                current_chunk = full_section
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append({
            'id': f'chunk_{chunk_id}',
            'content': current_chunk.strip(),
            'size': len(current_chunk),
            'type': 'section'
        })
    
    return chunks

def embed_text(text: str) -> List[float]:
    """Generate hybrid embedding (TF-IDF + Word2Vec) for the given text"""
    if tfidf_vectorizer is None or word2vec_model is None:
        raise ValueError("Embedding models not initialized. Call initialize_embedding_models() first.")

    tokens = preprocess_text(text)
    processed_text = " ".join(tokens)

    tfidf_vec = tfidf_vectorizer.transform([processed_text]).toarray()[0]
    vectors = [word2vec_model[w] for w in tokens if w in word2vec_model]
    w2v_vec = np.mean(vectors, axis=0) if vectors else np.zeros(W2V_DIM)

    hybrid = np.concatenate((tfidf_vec, w2v_vec))
    return hybrid.tolist()

def extract_and_process_pdf(pdf_bytes: bytes, chunk_size: int = 1000, overlap: int = 200) -> tuple:
    """
    Extract text from PDF using pdfplumber, clean it, and chunk it intelligently
    Returns: (cleaned_text, chunks, analysis)
    """
    print("🔧 Using pdfplumber for PDF extraction")
    
    # Extract text with pdfplumber
    raw_text = extract_text_with_pdfplumber(pdf_bytes)
    
    # Clean the text
    cleaned_text = clean_pdf_text(raw_text)
    
    # Analyze step order
    analysis = analyze_step_order(cleaned_text)
    
    # Chunk the text intelligently
    chunks = smart_chunk_text(cleaned_text, chunk_size, overlap)
    
    return cleaned_text, chunks, analysis

def process_and_store_pdf_in_chroma_improved(pdf_bytes: bytes, pdf_file_name: str, 
                                           chunk_size: int = 1000, overlap: int = 200,
                                           folder_id: int = None, document_type: str = 'other',
                                           target_program: str = 'all', keywords: str = ''):
    """
    Process PDF with improved pdfplumber extraction and store as ONE document per PDF in ChromaDB
    Enhanced with folder organization and metadata
    """
    print(f"🔄 Processing PDF: {pdf_file_name}")
    
    # Initialize embedding models
    initialize_embedding_models()
    
    # Get ChromaDB collection
    chroma_client = ChromaService.get_client()
    collection_name = os.getenv("CHROMA_COLLECTION", "documents")
    collection = chroma_client.get_or_create_collection(name=collection_name)
    print(f"✅ Using collection '{collection_name}'")
    
    # Extract and process PDF (we still analyze but don't chunk)
    try:
        raw_text = extract_text_with_pdfplumber(pdf_bytes)
        cleaned_text = clean_pdf_text(raw_text)
        analysis = analyze_step_order(cleaned_text)
        
        print(f"📊 Extracted {len(cleaned_text)} characters from PDF")
        
        # Log step analysis
        if analysis["found"]:
            print(f"🔍 Step analysis: {analysis['order']} ({len(analysis['steps'])} steps found)")
        else:
            print(f"🔍 Step analysis: {analysis['order']}")
            
    except Exception as e:
        print(f"❌ Failed to process PDF {pdf_file_name}: {e}")
        raise
    
    # Store as ONE document per PDF
    doc_id = os.path.splitext(pdf_file_name)[0]
    
    # Import Django models here to avoid circular imports
    from django.apps import apps
    DocumentFolder = apps.get_model('chatbot', 'DocumentFolder')
    DocumentMetadata = apps.get_model('chatbot', 'DocumentMetadata')
    
    try:
        # Get or create default folder if none specified
        if folder_id:
            try:
                folder = DocumentFolder.objects.get(id=folder_id)
            except DocumentFolder.DoesNotExist:
                folder, _ = DocumentFolder.objects.get_or_create(
                    name="Uncategorized",
                    defaults={'description': 'Default folder for uncategorized documents'}
                )
        else:
            folder, _ = DocumentFolder.objects.get_or_create(
                name="Uncategorized",
                defaults={'description': 'Default folder for uncategorized documents'}
            )
        
        # Create or update document metadata
        doc_metadata, created = DocumentMetadata.objects.update_or_create(
            document_id=doc_id,
            defaults={
                'filename': pdf_file_name,
                'folder': folder,
                'document_type': document_type,
                'target_program': target_program,
                'keywords': keywords,
                'synced_to_chroma': False,
            }
        )
        
        # Generate enhanced metadata for ChromaDB
        chroma_metadata = doc_metadata.get_chroma_metadata()
        chroma_metadata.update({
            'step_analysis': analysis["order"]
        })
        
        full_embedding = embed_text(cleaned_text)
        collection.add(
            ids=[doc_id],
            documents=[cleaned_text],
            embeddings=[full_embedding],
            metadatas=[chroma_metadata]
        )
        
        # Update sync status
        doc_metadata.synced_to_chroma = True
        doc_metadata.save()
        
        print(f"✅ Stored complete PDF as single document: {doc_id}")
        print(f"📁 Assigned to folder: {folder.name}")
        return 1  # Successfully stored 1 document
        
    except Exception as e:
        print(f"❌ Failed to store PDF {pdf_file_name}: {e}")
        raise

def sync_supabase_to_chroma_improved(chunk_size: int = 1000, overlap: int = 200) -> Dict:
    """
    Sync all PDFs from Supabase to ChromaDB with improved pdfplumber processing
    """
    print("🚀 Starting improved Supabase to ChromaDB sync (pdfplumber only)...")
    
    supabase = get_supabase_client()
    
    # List all PDFs
    try:
        files = supabase.storage.from_(PDF_BUCKET).list()
        pdf_names = [f['name'] for f in files if f['name'].lower().endswith('.pdf')]
        print(f"📄 Found {len(pdf_names)} PDF files")
    except Exception as e:
        print(f"❌ Failed to list files: {e}")
        return {"error": str(e), "found": 0, "ingested": 0}
    
    if not pdf_names:
        print("ℹ️ No PDF files found in bucket")
        return {"status": "ok", "found": 0, "ingested": 0}
    
    # Process each PDF
    ingested = 0
    failed = []
    
    for pdf_file in pdf_names:
        try:
            print(f"\n📥 Downloading: {pdf_file}")
            pdf_bytes = supabase.storage.from_(PDF_BUCKET).download(pdf_file)
            
            process_and_store_pdf_in_chroma_improved(pdf_bytes, pdf_file, chunk_size, overlap)
            ingested += 1
            print(f"✅ Successfully processed: {pdf_file}")
            
        except Exception as e:
            print(f"❌ Failed to process {pdf_file}: {e}")
            failed.append({"file": pdf_file, "error": str(e)})
            continue
    
    result = {
        "status": "ok", 
        "found": len(pdf_names), 
        "ingested": ingested,
        "failed": len(failed),
        "extraction_method": "pdfplumber"
    }
    
    if failed:
        result["failed_files"] = failed
    
    print(f"\n🎯 Sync complete: {ingested}/{len(pdf_names)} files processed successfully")
    return result

def main():
    """Main function for testing the improved sync"""
    print("🧪 Testing improved PDF sync (pdfplumber only)...")
    
    result = sync_supabase_to_chroma_improved()
    print("\n📋 Final Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
