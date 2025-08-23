"""
Fast hybrid chatbot that combines speed optimizations with TF-IDF and Word2Vec retrieval.
"""

import os
import sys
import numpy as np
import json
import time
from typing import List, Dict, Tuple, Optional, Generator
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

# Import the optimized interface - FIXED VERSION
try:
    from chatbot.llama_interface_optimized import (
        generate_response, 
        correct_typos, 
        stream_response, 
        llm, 
        GENERATION_CONFIG
    )
except ImportError:
    from llama_interface_optimized import (
        generate_response, 
        correct_typos, 
        stream_response, 
        llm, 
        GENERATION_CONFIG
    )

from chatbot.chroma_connection import ChromaService
from chatbot.test_pdf_to_chroma import initialize_embedding_models as _init_embed_models, embed_text as _embed_text

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
                 word2vec_path=None, use_chroma: bool = False, chroma_collection_name: Optional[str] = None):
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
        
        self.use_chroma = use_chroma
        self.chroma_collection_name = chroma_collection_name or os.getenv("CHROMA_COLLECTION", "documents")

        if self.use_chroma:
            _init_embed_models()  # ensure the TF‑IDF + Word2Vec embedder is ready
        else:
            self._load_data()
        
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
        if self.use_chroma:
            return self._retrieve_from_chroma(query, top_k)
        if self.vectors is not None and self.tfidf_vectorizer is not None:
            query_vector = self._vectorize_query(query)
            return self._vector_search(query_vector, top_k)
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
    
    def _retrieve_from_chroma(self, query: str, top_k: int = 2) -> List[Dict]:
        q_emb = _embed_text(query)
        try:
            collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)
            res = collection.query(
                query_embeddings=[q_emb],
                n_results=max(top_k, 5),
                include=["documents", "distances", "metadatas"],
                where={"source": "pdf_scrape"}  # only your Supabase-ingested PDFs
            )
            # relevance = 1/(1+d) fallback to 1.0 if no distances
            ids = res.get("ids", [[]])[0]
            docs = res.get("documents", [[]])[0]
            d = res.get("distances", [None])[0]
            out = []
            for i, (doc_id, content) in enumerate(zip(ids, docs)):
                if d is not None and i < len(d) and d[i] is not None:
                    rel = float(1.0 / (1.0 + d[i]))
                else:
                    rel = 1.0  # accept when distances aren’t present
                out.append({"id": doc_id, "content": content, "relevance": rel})
            return out
        except Exception as e:
            print(f"Error querying Chroma: {e}")
            return []
    
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
                      stream: bool = True, use_history: bool = True,
                      require_context: bool = True, min_relevance: float = 0.35) -> Tuple[str, List[Dict]]:
        # Start timing
        start_time = time.time()

        if stream:
            print("🔍 Searching for relevant information...", end="", flush=True)

        # Optional typo correction
        if correct_spelling and len(query) < 50:
            corrected_query = correct_typos(query)
            if corrected_query.lower() != query.lower():
                print(f"\rCorrected query: '{query}' → '{corrected_query}'")
                query = corrected_query

        # Retrieve docs
        try:
            relevant_docs = self.retrieve_documents(query)
        except Exception as e:
            print(f"\r❌ Retrieval error: {e}")
            relevant_docs = []

        # Relax the context filter so valid hits aren’t discarded
        filtered = []
        for d in relevant_docs:
            rel = d.get("relevance")
            try: rel = float(rel)
            except (TypeError, ValueError): rel = None
            if rel is None or rel >= min_relevance:
                filtered.append(d)

        # if nothing passes threshold but we do have hits, keep top-1
        if require_context and not filtered:
            if relevant_docs:
                filtered = [relevant_docs[0]]
            else:
                return "I don’t have enough information in my Admissions & Aid knowledge base to answer that.", []

        relevant_docs = filtered

        if require_context and not relevant_docs:
            return "I don’t have enough information in my Admissions & Aid knowledge base to answer that.", []

        retrieval_time = time.time() - start_time
        if stream:
            print("\r" + " " * 40 + "\r", end="", flush=True)
        print(f"⏱️ Document retrieval: {retrieval_time:.2f}s")

        # Build context from retrieved docs
        doc_context = "\n\n".join([
            f"Source: {doc.get('id','')}\n{doc['content'][:1000]}"
            for doc in relevant_docs[:3]
        ])

        # History context (optional, limited)
        history_context = ""
        if use_history and self.dialogue_history:
            history_context = "Previous conversation:\n"
            recent_history = self.dialogue_history[-2:] if len(self.dialogue_history) > 2 else self.dialogue_history
            for exchange in recent_history:
                history_context += f"User: {exchange['query']}\n"
                history_context += f"Assistant: {exchange['response'][:100]}...\n"

        # Strict instruction to avoid guessing
        prompt = f"""Context information:
            {doc_context}

            {history_context}
            Question: {query}

            Instructions: You must answer strictly and only using the context above.
            If the context does not contain enough information, reply exactly:
            "I don’t have enough information in my Admissions & Aid knowledge base to answer that."
            Answer:"""

        # Generate response
        if stream:
            response = stream_response(prompt, max_tokens=max_tokens)
        else:
            print("\n💬 Response: ", end="")
            response = generate_response(prompt, max_tokens=max_tokens)
            print(response)

        # Add to history
        self.add_to_history(query, response)

        total_time = time.time() - start_time
        print(f"⏱️ Total processing time: {total_time:.2f}s")

        return response, relevant_docs

    def process_query_stream(self, query: str, correct_spelling: bool = True, max_tokens: int = 150,
                        use_history: bool = True, require_context: bool = True, 
                        min_relevance: float = 0.35) -> Generator[Dict, None, None]:
        """
        Process query and yield streaming chunks for real-time response
        Yields dictionaries with 'chunk', 'error', or 'done' keys
        """
        try:
            # Start timing
            start_time = time.time()
            
            # Optional typo correction
            if correct_spelling and len(query) < 50:
                corrected_query = correct_typos(query)
                if corrected_query.lower() != query.lower():
                    query = corrected_query
                    yield {"info": f"Corrected query: '{query}' → '{corrected_query}'"}

            # Retrieve docs (same logic as process_query)
            try:
                relevant_docs = self.retrieve_documents(query)
            except Exception as e:
                yield {"error": f"Retrieval error: {e}"}
                return

            # Filter by relevance (same logic as process_query)
            filtered = []
            for d in relevant_docs:
                rel = d.get("relevance")
                try: 
                    rel = float(rel)
                except (TypeError, ValueError): 
                    rel = None
                if rel is None or rel >= min_relevance:
                    filtered.append(d)

            # Handle no context case
            if require_context and not filtered:
                if relevant_docs:
                    filtered = [relevant_docs[0]]
                else:
                    yield {"chunk": "I don't have enough information in my Admissions & Aid knowledge base to answer that."}
                    yield {"done": True}
                    return

            relevant_docs = filtered

            if require_context and not relevant_docs:
                yield {"chunk": "I don't have enough information in my Admissions & Aid knowledge base to answer that."}
                yield {"done": True}
                return

            # Build context from retrieved docs (same as process_query)
            doc_context = "\n\n".join([
                f"Source: {doc.get('id','')}\n{doc['content'][:1000]}"
                for doc in relevant_docs[:3]
            ])

            # History context (same as process_query)
            history_context = ""
            if use_history and self.dialogue_history:
                # Calculate available token budget for history
                base_prompt = f"Context information:\n{doc_context}\nQuestion: {query}\nInstructions: You must answer strictly and only using the context above.\nAnswer:"
                base_tokens = len(base_prompt.split())
                
                # Leave 300 tokens for response generation, use rest for history
                available_for_history = 3700 - base_tokens
                
                if available_for_history > 80:  # Only add history if meaningful space
                    history_context = "Previous conversation:\n"
                    
                    # Start with most recent exchange and work backwards
                    for exchange in reversed(self.dialogue_history):
                        # Create a condensed version of the exchange
                        condensed_q = exchange['query'][:50] + "..." if len(exchange['query']) > 50 else exchange['query']
                        condensed_a = exchange['response'][:60] + "..." if len(exchange['response']) > 60 else exchange['response']
                        
                        entry = f"User: {condensed_q}\nAssistant: {condensed_a}\n"
                        entry_tokens = len(entry.split())
                        
                        # Check if we have room for this entry
                        if len((history_context + entry).split()) <= available_for_history:
                            history_context += entry
                        else:
                            break  # Stop adding history if we run out of space
                    
                    # If no history could fit, don't include any
                    if history_context == "Previous conversation:\n":
                        history_context = ""

            # Build prompt (same as process_query)
            prompt = f"""Context information:
{doc_context}

{history_context}
Question: {query}

Instructions: You must answer strictly and only using the context above.
If the context does not contain enough information, reply exactly:
"I don't have enough information in my Admissions & Aid knowledge base to answer that."
Answer:"""

            print(f"🔍 DEBUG INFO:")
            print(f"📊 Query: '{query}'")
            print(f"📄 Retrieved docs: {len(relevant_docs)}")
            for i, doc in enumerate(relevant_docs):
                print(f"   Doc {i+1}: {doc['id']} (Relevance: {doc.get('relevance', 'N/A'):.3f})")
            print(f"💬 History entries: {len(self.dialogue_history)}")
            print(f"📏 Doc context: {len(doc_context)} chars (~{len(doc_context.split())} tokens)")
            print(f"📏 History context: {len(history_context)} chars (~{len(history_context.split())} tokens)")
            print(f"📏 Total prompt: {len(prompt)} chars (~{len(prompt.split())} tokens)")
            print(f"📏 Estimated tokens left for response: {4096 - len(prompt.split())}")
            print("=" * 70)
            
            # Stream from LLM
            full_response = ""
            
            # Create streaming config
            stream_config = GENERATION_CONFIG.copy()
            stream_config.update({
                "stream": True,
                "max_tokens": max_tokens
            })
            
            try:
                # Generate streaming response from LLM
                for chunk in llm(prompt, echo=False, **stream_config):
                    if isinstance(chunk, dict):
                        # Handle llama-cpp-python response format
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            choice = chunk['choices'][0]
                            
                            # Handle different response formats
                            text_chunk = ""
                            if 'delta' in choice and 'content' in choice['delta']:
                                text_chunk = choice['delta']['content']
                            elif 'text' in choice:
                                text_chunk = choice['text']
                            
                            if text_chunk:
                                full_response += text_chunk
                                yield {"chunk": text_chunk}
                
                # Add to conversation history
                self.add_to_history(query, full_response)
                
                # Calculate total time
                total_time = time.time() - start_time
                yield {"info": f"Total processing time: {total_time:.2f}s"}
                yield {"done": True}
                
            except Exception as e:
                yield {"error": f"LLM generation error: {e}"}
                
        except Exception as e:
            yield {"error": f"Processing error: {e}"}

def test_fast_hybrid_chatbot():
    """Test the fast hybrid chatbot"""
    chatbot = FastHybridChatbot(use_chroma=True)
    
    print("\n⚡🔍 FAST HYBRID CHATBOT MODE ⚡🔍")
    print("Type 'exit' to quit\n")
    # No more options for toggling streaming/history

    max_tokens = 500  # Higher default for complete responses
    streaming = True  # Always streaming
    use_history = True  # Always use history

    while True:
        query = input("🔍 Query: ")
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("👋 Goodbye!")
            break
        elif query.lower() == 'clear history':
            chatbot.clear_history()
            continue
        
        # Always use streaming and history
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
        print("-" * 80)

if __name__ == "__main__":
    test_fast_hybrid_chatbot() 