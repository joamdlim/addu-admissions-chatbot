"""
Fast hybrid chatbot that combines speed optimizations with TF-IDF and Word2Vec retrieval.
"""

import os
import sys
import numpy as np
import json
import time
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import preprocess
try:
    from chatbot.preprocess import preprocess_text
except ImportError:
    # Simplified preprocessing if import fails
    def preprocess_text(text):
        return text.lower().split()

# Try to import Word2Vec
try:
    import gensim
    from gensim.models import KeyedVectors
    WORD2VEC_AVAILABLE = True
    print("✅ Gensim Word2Vec available")
except ImportError:
    WORD2VEC_AVAILABLE = False
    print("⚠️ Gensim Word2Vec not available, using placeholder")

# Import the optimized interface
try:
    from chatbot.llama_interface_optimized import generate_response, correct_typos, stream_response
except ImportError:
    from chatbot.llama_interface_optimized import generate_response, correct_typos, stream_response

def sparse_to_array(sparse_matrix):
    """Convert sparse matrix to numpy array safely"""
    if hasattr(sparse_matrix, "toarray"):
        return sparse_matrix.toarray()
    elif isinstance(sparse_matrix, np.ndarray):
        return sparse_matrix
    else:
        return np.array(sparse_matrix)

def compute_word2vec_vector(tokens, model=None, dim=300):
    """Compute Word2Vec vector for tokens, with fallback to placeholder"""
    if not tokens:
        return np.zeros(dim)
    
    if model is not None:
        # Use actual Word2Vec model if available
        vectors = []
        for token in tokens:
            try:
                if token in model:
                    vectors.append(model[token])
            except:
                pass
        
        if vectors:
            # Average the vectors
            return np.mean(vectors, axis=0)
    
    # Fallback to placeholder
    return np.zeros(dim)

class FastHybridChatbot:
    """Fast hybrid chatbot that combines TF-IDF, Word2Vec, and optimized LLM"""
    
    def __init__(self, embeddings_dir=None, processed_dir=None, 
                 word2vec_path=None):
        # Get the current script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set default paths relative to the current script
        if embeddings_dir is None:
            embeddings_dir = os.path.join(current_dir, "embeddings")
        if processed_dir is None:
            processed_dir = os.path.join(current_dir, "processed")
        if word2vec_path is None:
            word2vec_path = os.path.join(current_dir, "model", "GoogleNews-vectors-negative300.bin")
        
        self.vectors_path = os.path.join(embeddings_dir, "hybrid_vectors.npy")
        self.metadata_path = os.path.join(embeddings_dir, "metadata.json")
        
        # Debug information
        print(f"Looking for metadata at: {self.metadata_path}")
        print(f"Looking for vectors at: {self.vectors_path}")
        
        self.documents = []
        self.vectors = None
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        # Initialize dialogue history
        self.dialogue_history = []
        self.max_history_length = 5  # Keep last 5 exchanges
        
        # Start timing
        start_time = time.time()
        
        # Try to load Word2Vec model if available
        if WORD2VEC_AVAILABLE and os.path.exists(word2vec_path):
            try:
                print(f"🔄 Loading Word2Vec model from {word2vec_path}")
                self.word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
                print("✅ Word2Vec model loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load Word2Vec model: {e}")
        
        # Load data
        self._load_data()
        
        # Report load time
        load_time = time.time() - start_time
        print(f"⚡ Fast hybrid chatbot initialized in {load_time:.2f} seconds")
    
    def _load_data(self):
        """Load vectors and metadata with optimizations"""
        try:
            # Load metadata
            print(f"Attempting to load metadata from: {self.metadata_path}")
            print(f"Current working directory: {os.getcwd()}")
            
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                print(f"✅ Loaded metadata for {len(self.documents)} documents")
            else:
                print(f"⚠️ Metadata file not found at: {self.metadata_path}")
                # Try alternative locations
                alt_paths = [
                    os.path.join(os.getcwd(), "embeddings", "metadata.json"),
                    os.path.join(os.path.dirname(os.getcwd()), "embeddings", "metadata.json"),
                    os.path.join(os.getcwd(), "backend", "embeddings", "metadata.json")
                ]
                
                for path in alt_paths:
                    print(f"Trying alternative path: {path}")
                    if os.path.exists(path):
                        print(f"Found metadata at: {path}")
                        with open(path, 'r', encoding='utf-8') as f:
                            self.documents = json.load(f)
                        print(f"✅ Loaded metadata for {len(self.documents)} documents")
                        self.metadata_path = path  # Update the path
                        break
                else:
                    print("❌ Could not find metadata.json in any location")
                    return
            
            # Load vectors if they exist
            if os.path.exists(self.vectors_path):
                self.vectors = np.load(self.vectors_path)
                print(f"✅ Loaded vectors with shape: {self.vectors.shape}")
                
                # Build TF-IDF vectorizer on document content
                corpus = [" ".join(preprocess_text(doc["content"])) for doc in self.documents]
                self.tfidf_vectorizer = TfidfVectorizer()
                self.tfidf_vectorizer.fit(corpus)
                print("✅ Built TF-IDF vectorizer")
            else:
                print("⚠️ Vector file not found, falling back to keyword search")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def _vectorize_query(self, query: str) -> np.ndarray:
        """Convert query to vector representation - optimized for speed"""
        # Simple preprocessing
        processed_query = preprocess_text(query)
        query_text = " ".join(processed_query)
        
        # Get TF-IDF vector
        try:
            if self.tfidf_vectorizer is None:
                print("⚠️ TF-IDF vectorizer not available")
                # Estimate a reasonable dimension for TF-IDF
                tfidf_dim = 1000 if self.vectors is None else self.vectors.shape[1] - 300
                tfidf_vector = np.zeros(tfidf_dim)
            else:
                tfidf_vector = sparse_to_array(self.tfidf_vectorizer.transform([query_text]))[0]
        except Exception as e:
            print(f"❌ Error creating TF-IDF vector: {e}")
            # Create a zero vector with a reasonable dimension
            tfidf_dim = 1000 if self.vectors is None else self.vectors.shape[1] - 300
            tfidf_vector = np.zeros(tfidf_dim)
        
        # Get Word2Vec vector - use model if available
        w2v_vector = compute_word2vec_vector(processed_query, self.word2vec_model)
        
        # Combine them
        hybrid_vector = np.concatenate((tfidf_vector, w2v_vector))
        
        return hybrid_vector
    
    def retrieve_documents(self, query: str, top_k: int = 2) -> List[Dict]:
        """Retrieve relevant documents using hybrid approach or fallback to keyword search"""
        if self.vectors is not None and self.tfidf_vectorizer is not None:
            # Use vector similarity if vectors are available
            query_vector = self._vectorize_query(query)
            return self._vector_search(query_vector, top_k)
        else:
            # Fall back to keyword search
            return self._keyword_search(query, top_k)
    
    def _vector_search(self, query_vector: np.ndarray, top_k: int = 2) -> List[Dict]:
        """Find relevant documents using vector similarity - optimized for speed"""
        try:
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector.reshape(1, -1), self.vectors)[0]
            
            # Get top K matches using argpartition (faster than argsort)
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            
            # Create result list
            relevant_docs = []
            for idx in top_indices:
                similarity_score = similarities[idx]
                if similarity_score > 0.001:  # Very low threshold
                    doc = {
                        'id': self.documents[idx]['id'],
                        'content': self.documents[idx]['content'],
                        'relevance': float(similarity_score)
                    }
                    relevant_docs.append(doc)
            
            # Sort by relevance
            return sorted(relevant_docs, key=lambda x: x['relevance'], reverse=True)
        except Exception as e:
            print(f"❌ Error in vector search: {e}")
            return []
    
    def _keyword_search(self, query: str, max_docs: int = 2) -> List[Dict]:
        """Simple keyword search as fallback"""
        if not self.documents:
            return []
        
        # Convert query to lowercase for case-insensitive matching
        query_terms = query.lower().split()
        
        # Score documents based on keyword matches
        scored_docs = []
        for doc in self.documents:
            content = doc['content'].lower()
            score = sum(content.count(term) for term in query_terms)
            
            if score > 0:
                scored_docs.append({
                    'id': doc['id'],
                    'content': doc['content'],
                    'relevance': score
                })
        
        # Sort by relevance and take top results
        return sorted(scored_docs, key=lambda x: x['relevance'], reverse=True)[:max_docs]
    
    def add_to_history(self, query: str, response: str) -> None:
        """Add a query-response pair to dialogue history"""
        self.dialogue_history.append({"query": query, "response": response})
        # Keep history within max length
        if len(self.dialogue_history) > self.max_history_length:
            self.dialogue_history.pop(0)
    
    def clear_history(self) -> None:
        """Clear dialogue history"""
        self.dialogue_history = []
        print("🧹 Dialogue history cleared")
    
    def get_context_from_history(self) -> str:
        """Format dialogue history as context for the LLM"""
        if not self.dialogue_history:
            return ""
        
        context = "Previous conversation:\n"
        for i, exchange in enumerate(self.dialogue_history):
            context += f"User: {exchange['query']}\n"
            context += f"Assistant: {exchange['response']}\n"
        
        return context
    
    def process_query(self, query: str, correct_spelling: bool = True, max_tokens: int = 150, 
                      stream: bool = True, use_history: bool = True) -> Tuple[str, List[Dict]]:
        """Process a query with hybrid retrieval and fast response generation"""
        # Start timing
        start_time = time.time()
        
        # Start showing that we're working immediately
        if stream:
            print("🔍 Searching for relevant information...", end="", flush=True)
        
        # Correct typos if enabled - but only if it's a short query to avoid delays
        if correct_spelling and len(query) < 50:
            corrected_query = correct_typos(query)
            if corrected_query.lower() != query.lower():
                print(f"\rCorrected query: '{query}' → '{corrected_query}'")
                query = corrected_query
        
        # Limit retrieval time to ensure quick response
        retrieval_start = time.time()
        max_retrieval_time = 2.0  # Maximum 2 seconds for retrieval
        
        # Retrieve relevant documents with time limit
        try:
            relevant_docs = self.retrieve_documents(query)
            
            # If retrieval is taking too long, proceed with what we have
            if time.time() - retrieval_start > max_retrieval_time and relevant_docs:
                print("\r⚠️ Retrieval time limited, proceeding with partial results")
        except Exception as e:
            print(f"\r❌ Retrieval error: {e}")
            relevant_docs = []
        
        retrieval_time = time.time() - start_time
        
        # Clear the searching message if streaming
        if stream:
            print("\r" + " " * 40 + "\r", end="", flush=True)
        
        print(f"⏱️ Document retrieval: {retrieval_time:.2f}s")
        
        # Format context - optimized for speed with limited content
        doc_context = "\n".join([f"Document: {doc['content'][:150]}" for doc in relevant_docs[:2]])
        
        # Add dialogue history context if available and enabled
        # But limit the history to just the most recent exchanges for speed
        history_context = ""
        if use_history and self.dialogue_history:
            history_context = "Previous conversation:\n"
            # Only use the last 2 exchanges for faster response
            recent_history = self.dialogue_history[-2:] if len(self.dialogue_history) > 2 else self.dialogue_history
            for exchange in recent_history:
                history_context += f"User: {exchange['query']}\n"
                history_context += f"Assistant: {exchange['response'][:100]}...\n"
        
        # Create improved prompt that encourages complete responses
        prompt = f"""Context information:
{doc_context}

{history_context}
Question: {query}

Instructions: Provide a complete and comprehensive answer to the question based on the context information.
If the question relates to previous conversation, use that context to provide a relevant answer.
If the context doesn't contain enough information, say so but provide what you can.
Answer:"""
        
        # Generate response with streaming or non-streaming based on parameter
        if stream:
            # Use streaming response generation - this always returns a string
            response = stream_response(prompt, max_tokens=max_tokens)
        else:
            # Show that we're generating a response
            print("\n💬 Response: ", end="")
            
            # Generate response with increased max_tokens - use the function that guarantees a string
            response = generate_response(prompt, max_tokens=max_tokens)
            
            # Print the response
            print(response)
        
        # Add to dialogue history
        self.add_to_history(query, response)
        
        # Calculate total time
        total_time = time.time() - start_time
        print(f"⏱️ Total processing time: {total_time:.2f}s")
        
        return response, relevant_docs

def test_fast_hybrid_chatbot():
    """Test the fast hybrid chatbot"""
    chatbot = FastHybridChatbot()
    
    print("\n⚡🔍 FAST HYBRID CHATBOT MODE ⚡🔍")
    print("Type 'exit' to quit\n")
    print("Options:")
    print("  - Type 'stream on' to enable streaming responses")
    print("  - Type 'stream off' to disable streaming responses")
    print("  - Type 'history on' to enable dialogue history")
    print("  - Type 'history off' to disable dialogue history")
    print("  - Type 'clear history' to clear dialogue history")
    
    # Set a generous max_tokens value to ensure complete responses
    max_tokens = 500  # Higher default for complete responses
    streaming = True  # Default to streaming
    use_history = True  # Default to using history
    
    while True:
        query = input("🔍 Query: ")
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("👋 Goodbye!")
            break
        elif query.lower() == 'stream on':
            streaming = True
            print("Streaming responses enabled")
            continue
        elif query.lower() == 'stream off':
            streaming = False
            print("Streaming responses disabled")
            continue
        elif query.lower() == 'history on':
            use_history = True
            print("Dialogue history enabled")
            continue
        elif query.lower() == 'history off':
            use_history = False
            print("Dialogue history disabled")
            continue
        elif query.lower() == 'clear history':
            chatbot.clear_history()
            continue
        
        # Process query with current settings
        response, relevant_docs = chatbot.process_query(
            query, 
            max_tokens=max_tokens,
            stream=streaming,
            use_history=use_history
        )
        
        # Show document info
        print(f"\n📚 Retrieved {len(relevant_docs)} document(s)")
        for i, doc in enumerate(relevant_docs):
            print(f"  [{i+1}] {doc['id']} (Relevance: {doc['relevance']:.4f})")
        
        # Response is already printed in streaming mode
        if not streaming:
            print("\n")
        
        print("-" * 80)

if __name__ == "__main__":
    test_fast_hybrid_chatbot() 