# backend/chatbot/test_hybrid_chatbot.py
import os
import sys
import numpy as np
import json
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import time

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import preprocess directly
from preprocess import preprocess_text

# Make a modified version of the needed functions from vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

def build_tfidf_vectorizer(corpus):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    return vectorizer

def sparse_to_array(sparse_matrix):
    """Convert sparse matrix to numpy array safely"""
    if hasattr(sparse_matrix, "toarray"):
        return sparse_matrix.toarray()
    elif isinstance(sparse_matrix, np.ndarray):
        return sparse_matrix
    else:
        # Handle other types or raise an appropriate error
        return np.array(sparse_matrix)

def compute_word2vec_vector(tokens):
    # Simple placeholder since we're just testing
    return np.zeros(300)

# Import from llama_interface
from chatbot.llama_interface import llm, correct_typos, extract_response_text
try:
    from chatbot.llama_config import GENERATION_CONFIG
except ImportError:
    from llama_config import GENERATION_CONFIG

class TestHybridChatbot:
    def __init__(self, embeddings_dir="../embeddings", processed_dir="../processed"):
        self.vectors_path = os.path.join(embeddings_dir, "hybrid_vectors.npy")
        self.metadata_path = os.path.join(embeddings_dir, "metadata.json")
        self.processed_dir = processed_dir
        
        # Load vectors and metadata
        print("🔄 Loading vectors and metadata...")
        self.vectors = None
        self.documents = []
        self._load_vectors_and_metadata()
        
        # Initialize TFIDF vectorizer from documents
        if self.documents:
            corpus = [" ".join(preprocess_text(doc["content"])) for doc in self.documents]
            self.tfidf_vectorizer = build_tfidf_vectorizer(corpus)
            print(f"✅ TFIDF Vectorizer built on {len(corpus)} documents")
        else:
            print("⚠️ No documents to build TFIDF vectorizer")
    
    def _load_vectors_and_metadata(self):
        """Load the vectors and metadata from disk"""
        try:
            # Load metadata first
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                print(f"✅ Loaded metadata for {len(self.documents)} documents")
            else:
                print("⚠️ Metadata file not found")
                return
            
            # Always regenerate vectors for testing
            print("🔄 Generating vectors from documents...")
            # Extract content and preprocess
            corpus = [" ".join(preprocess_text(doc["content"])) for doc in self.documents]
            
            # Build TF-IDF vectors
            self.tfidf_vectorizer = build_tfidf_vectorizer(corpus)
            tfidf_vectors = sparse_to_array(self.tfidf_vectorizer.transform(corpus))
            print(f"✅ TFIDF vectors shape: {tfidf_vectors.shape}")
            
            # Generate Word2Vec vectors (placeholders in this case)
            w2v_vectors = [compute_word2vec_vector(preprocess_text(doc["content"])) for doc in self.documents]
            print(f"✅ Word2Vec vectors shape: ({len(w2v_vectors)}, {len(w2v_vectors[0])})")
            
            # Combine them
            self.vectors = np.array([np.concatenate((tfidf_vec, w2v_vec)) for tfidf_vec, w2v_vec in zip(tfidf_vectors, w2v_vectors)])
            print(f"✅ Generated vectors with shape: {self.vectors.shape}")
            
            # Save the vectors for future use
            try:
                np.save(self.vectors_path, self.vectors)
                print(f"✅ Saved vectors to {self.vectors_path}")
            except Exception as e:
                print(f"⚠️ Could not save vectors: {e}")
                
        except Exception as e:
            print(f"❌ Error in vector/metadata processing: {e}")
    
    def _vectorize_query(self, query: str) -> np.ndarray:
        """Convert user query to the same hybrid vector space as documents"""
        # Simplified preprocessing - skip full preprocessing for speed
        processed_query = query.lower().split()
        query_text = " ".join(processed_query)
        
        # Get TFIDF vector - with error handling for speed
        try:
            tfidf_vector = sparse_to_array(self.tfidf_vectorizer.transform([query_text]))[0]
        except Exception as e:
            print(f"❌ Error creating TFIDF vector: {e}")
            # Create a zero vector with the right dimension
            tfidf_vector = np.zeros(self.tfidf_vectorizer.get_feature_names_out().shape[0])
        
        # Get Word2Vec vector - simplified for speed
        w2v_vector = np.zeros(300)  # Skip actual computation for speed
        
        # Combine them
        hybrid_vector = np.concatenate((tfidf_vector, w2v_vector))
        
        return hybrid_vector
    
    def _retrieve_relevant_docs(self, query_vector: np.ndarray, top_k: int = 2) -> List[Dict]:
        """Find most relevant documents using cosine similarity"""
        if self.vectors is None or len(self.documents) == 0:
            print("⚠️ No vectors or documents available")
            return []
        
        try:
            # Calculate cosine similarity - use optimized version
            similarities = cosine_similarity(query_vector.reshape(1, -1), self.vectors)[0]
            
            # Get top K matches directly without sorting all
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            
            relevant_docs = []
            for idx in top_indices:
                similarity_score = similarities[idx]
                # Lower threshold further for testing
                if similarity_score > 0.001:
                    doc = {
                        'id': self.documents[idx]['id'],
                        'content': self.documents[idx]['content'],
                        'relevance': float(similarity_score)
                    }
                    relevant_docs.append(doc)
                
            return relevant_docs
        except Exception as e:
            print(f"❌ Error calculating document similarity: {e}")
            return []
    
    def _generate_response(self, query: str, relevant_docs: List[Dict]) -> str:
        """Generate a response using Llama and retrieved documents"""
        # Format retrieved context - limit to much shorter excerpts for speed
        # Only include the most relevant parts of documents and limit to top 2 docs
        top_docs = sorted(relevant_docs, key=lambda x: x.get('relevance', 0), reverse=True)[:2]
        
        # Extract only the first 150 characters from each document
        context = "\n".join([f"Doc: {doc['content'][:150]}..." for doc in top_docs])
        
        # Create a much more concise prompt for Llama
        prompt = f"""Answer this question based on these docs:
{context}

Question: {query}
Answer:"""
        
        # Generate response using the Llama model with optimized parameters
        try:
            # Use the generation config from our config file
            result = llm(prompt, echo=False, **GENERATION_CONFIG)
            
            # Use our helper function to extract the response
            return extract_response_text(result)
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "I'm sorry, I had trouble generating a response. Please try again."
    
    def process_query(self, query: str) -> Tuple[str, List[Dict]]:
        """Main method to process a user query and return a response"""
        # Skip typo correction for speed unless specifically needed
        # corrected_query = correct_typos(query)
        corrected_query = query  # Skip correction for speed
        
        # If query was corrected, note that
        query_info = ""
        if corrected_query.lower() != query.lower():
            query_info = f"Corrected query: {corrected_query}\n"
            query = corrected_query
        
        # Vectorize the query
        query_vector = self._vectorize_query(query)
        
        # Retrieve relevant documents - reduce to top 2 for speed
        relevant_docs = self._retrieve_relevant_docs(query_vector, top_k=2)
        
        # Generate response
        response = self._generate_response(query, relevant_docs)
        
        return response, relevant_docs

# Create a test function to demo the hybrid chatbot
def test_hybrid_chatbot_interactive():
    chatbot = TestHybridChatbot()
    
    print("\n🤖 HYBRID CHATBOT INTERACTIVE MODE (OPTIMIZED FOR SPEED) 🤖")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("🔍 Query: ")
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("👋 Goodbye!")
            break
        
        # Start timing
        start_time = time.time()
        
        response, relevant_docs = chatbot.process_query(query)
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        
        print(f"\n📚 Retrieved {len(relevant_docs)} document(s) in {elapsed:.2f} seconds")
        
        # Only show document IDs and relevance scores for speed
        for i, doc in enumerate(relevant_docs):
            print(f"  [{i+1}] {doc['id']} (Relevance: {doc['relevance']:.2f})")
        
        print("\n💬 Response: ", end="")
        print(response)  # Print immediately without typing effect
        
        print("\n")
        print("-" * 80)

if __name__ == "__main__":
    # test_hybrid_chatbot()  # Comment out or remove the original test function
    test_hybrid_chatbot_interactive()