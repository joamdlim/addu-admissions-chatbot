"""
Fast hybrid chatbot that combines speed optimizations with TF-IDF and Word2Vec retrieval.
Updated to use Together AI instead of local GGUF model.
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

# Import the Together AI interface instead of llama_interface_optimized
try:
    from chatbot.together_ai_interface import (
        generate_response, 
        correct_typos, 
        stream_response, 
        llm, 
        TOGETHER_CONFIG
    )
except ImportError:
    from together_ai_interface import (
        generate_response, 
        correct_typos, 
        stream_response, 
        llm, 
        TOGETHER_CONFIG
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

class FastHybridChatbotTogether:
    """Fast hybrid chatbot that combines TF-IDF, Word2Vec, and Together AI"""
    
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

        # Dynamic program detection cache
        self._program_cache = {}
        self._filename_patterns_cache = {}
        self._last_cache_update = 0

        if self.use_chroma:
            _init_embed_models()  # ensure the TF‑IDF + Word2Vec embedder is ready
            self._discover_programs_from_data()  # Initialize dynamic program detection
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
        print(f"⚡ Fast hybrid chatbot with Together AI initialized in {load_time:.2f} seconds")

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
    
    def retrieve_documents_hybrid(self, query: str, top_k: int = 2) -> List[Dict]:
        """Multi-stage hybrid retrieval using semantic + metadata + folder intelligence + dynamic program detection"""
        print(f"🔍 HYBRID RETRIEVAL for: '{query}'")
        
        # Stage 1: Enhanced Intent Analysis with Dynamic Detection
        query_lower = query.lower()
        
        # Detect if this is a curriculum/subject query
        is_curriculum_query = any(word in query_lower for word in [
            'curriculum', 'course', 'subject', 'program structure', 'study plan',
            'what courses', 'what subjects', 'semester courses', 'year courses'
        ])
        
        # Enhanced detection with dynamic program discovery
        folder_and_filename = self._detect_smart_folder_and_filename_dynamic(query_lower, None)
        document_type, type_confidence = self._detect_document_type_with_confidence(query_lower)
        program_filter, program_confidence = self._detect_program_with_confidence(query_lower)
        
        # Override document type for curriculum queries
        if is_curriculum_query:
            document_type = 'academic'
            folder_and_filename['folder_filter'] = 'Subjects'
        
        intent = {
            'document_type': document_type,
            'program_filter': program_filter,
            'folder_filter': folder_and_filename.get('folder_filter'),
            'program_hint': folder_and_filename.get('program_hint'),
            'is_curriculum_query': is_curriculum_query,
            'type_confidence': type_confidence,
            'program_confidence': program_confidence
        }
        
        print(f"📊 Intent: folder={intent.get('folder_filter')}, program_hint={intent.get('program_hint')}, type={document_type}")
        
        # Stage 2: Multi-Strategy Retrieval
        all_candidates = []
        
        # Strategy A: Filename-Aware Retrieval (Highest Priority for Subjects)
        if intent.get('folder_filter') == 'Subjects' and intent.get('program_hint'):
            filename_results = self._retrieve_with_filename_intelligence_dynamic(query, intent, top_k)
            all_candidates.extend([(doc, 'filename', doc['relevance'] * 1.3) for doc in filename_results])
            print(f"📄 Filename strategy: {len(filename_results)} docs")
        
        # Strategy B: Metadata-First (High Precision)
        if document_type or program_filter:
            metadata_results = self._retrieve_with_metadata_priority(query, intent, top_k)
            all_candidates.extend([(doc, 'metadata', doc['relevance'] * 1.2) for doc in metadata_results])
            print(f"📋 Metadata strategy: {len(metadata_results)} docs")
        
        # Strategy C: Folder-Aware (Medium Precision)
        if intent.get('folder_filter'):
            folder_results = self._retrieve_with_folder_priority(query, intent, top_k)
            all_candidates.extend([(doc, 'folder', doc['relevance'] * 1.1) for doc in folder_results])
            print(f"📁 Folder strategy: {len(folder_results)} docs")
        
        # Strategy D: Pure Semantic (High Recall)
        semantic_results = self._retrieve_semantic_only(query, top_k * 2)
        all_candidates.extend([(doc, 'semantic', doc['relevance']) for doc in semantic_results])
        print(f"🧠 Semantic strategy: {len(semantic_results)} docs")
        
        # Stage 3: Smart Deduplication & Scoring
        final_results = self._merge_and_rank_candidates(all_candidates, query, top_k)
        
        print(f"✅ Final results: {len(final_results)} docs")
        for i, doc in enumerate(final_results):
            filename = doc.get('filename', 'N/A')
            strategy = doc.get('retrieval_strategy', 'N/A')
            score = doc.get('hybrid_score', 0)
            program_hint = intent.get('program_hint', 'N/A')
            print(f"   {i+1}. {filename} (strategy: {strategy}, score: {score:.3f}, program: {program_hint})")
        
        return final_results

    def retrieve_documents(self, query: str, top_k: int = 2) -> List[Dict]:
        """Main retrieval method - use hybrid approach with dynamic program detection"""
        if self.use_chroma:
            return self.retrieve_documents_hybrid(query, top_k)
        elif self.vectors is not None and self.tfidf_vectorizer is not None:
            query_vector = self._vectorize_query(query)
            return self._vector_search(query_vector, top_k)
        else:
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
    
    def _retrieve_from_chroma(self, query: str, top_k: int = 2, 
                             folder_filter: str = None, document_type_filter: str = None,
                             program_filter: str = None) -> List[Dict]:
        q_emb = _embed_text(query)
        try:
            collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)
            
            # Build where clause for filtering
            where_clause = {"source": "pdf_scrape"}  # Base filter
            
            # Add folder filtering
            if folder_filter:
                where_clause["folder_name"] = folder_filter
            
            # Add document type filtering
            if document_type_filter:
                where_clause["document_type"] = document_type_filter
                
            # Add program filtering
            if program_filter and program_filter != 'all':
                where_clause["target_program"] = {"$in": [program_filter, "all"]}
            
            res = collection.query(
                query_embeddings=[q_emb],
                n_results=max(top_k, 5),
                include=["documents", "distances", "metadatas"],
                where=where_clause
            )
            
            # relevance = 1/(1+d) fallback to 1.0 if no distances
            ids = res.get("ids", [[]])[0]
            docs = res.get("documents", [[]])[0]
            metadatas = res.get("metadatas", [[]])[0]
            d = res.get("distances", [None])[0]
            out = []
            for i, (doc_id, content) in enumerate(zip(ids, docs)):
                if d is not None and i < len(d) and d[i] is not None:
                    rel = float(1.0 / (1.0 + d[i]))
                else:
                    rel = 1.0  # accept when distances aren't present
                    
                # Include metadata in result
                metadata = metadatas[i] if i < len(metadatas) else {}
                out.append({
                    "id": doc_id, 
                    "content": content, 
                    "relevance": rel,
                    "folder": metadata.get("folder_name", "Unknown"),
                    "document_type": metadata.get("document_type", "other"),
                    "target_program": metadata.get("target_program", "all"),
                    "filename": metadata.get("filename", "")
                })
            return out
        except Exception as e:
            print(f"Error querying Chroma: {e}")
            return []

    def analyze_query_intent(self, query: str) -> Dict[str, str]:
        """Analyze query to determine appropriate filters"""
        query_lower = query.lower()
        
        # Document type detection
        document_type = None
        if any(word in query_lower for word in ['admission', 'apply', 'requirement', 'entrance']):
            document_type = 'admission'
        elif any(word in query_lower for word in ['enroll', 'registration', 'process']):
            document_type = 'enrollment'
        elif any(word in query_lower for word in ['scholarship', 'financial aid', 'grant']):
            document_type = 'scholarship'
        elif any(word in query_lower for word in ['program', 'course', 'degree', 'major']):
            document_type = 'academic'
        elif any(word in query_lower for word in ['fee', 'cost', 'payment', 'tuition']):
            document_type = 'fees'
        elif any(word in query_lower for word in ['contact', 'phone', 'email', 'office']):
            document_type = 'contact'
        
        # Program level detection
        program_filter = None
        if any(word in query_lower for word in ['undergraduate', 'bachelor', 'college']):
            program_filter = 'undergraduate'
        elif any(word in query_lower for word in ['graduate', 'master', 'phd', 'doctoral']):
            program_filter = 'graduate'
        elif any(word in query_lower for word in ['senior high', 'shs', 'grade 11', 'grade 12']):
            program_filter = 'senior_high'
        
        return {
            'document_type': document_type,
            'program_filter': program_filter,
            'folder_filter': None  # Can be extended for folder-specific queries
        }

    def process_query_with_intent_analysis(self, query: str, correct_spelling: bool = True, 
                                          max_tokens: int = 150, stream: bool = True, 
                                          use_history: bool = True, require_context: bool = True, 
                                          min_relevance: float = 0.35, manual_filters: Dict = None) -> Tuple[str, List[Dict]]:
        """Enhanced query processing with intent analysis and filtering"""
        start_time = time.time()

        if stream:
            print("🔍 Analyzing query intent...", end="", flush=True)

        # Determine filters from query intent or use manual overrides
        if manual_filters:
            filters = manual_filters
        else:
            filters = self.analyze_query_intent(query)
        
        if stream and any(filters.values()):
            filter_info = []
            if filters.get('document_type'):
                filter_info.append(f"type:{filters['document_type']}")
            if filters.get('program_filter'):
                filter_info.append(f"program:{filters['program_filter']}")
            print(f"\r🎯 Query filters: {', '.join(filter_info)}", end="", flush=True)

        # Optional typo correction
        if correct_spelling and len(query) < 50:
            corrected_query = correct_typos(query)
            if corrected_query.lower() != query.lower():
                print(f"\rCorrected query: '{query}' → '{corrected_query}'")
                query = corrected_query

        # Retrieve docs with filters
        try:
            relevant_docs = self._retrieve_from_chroma(
                query, 
                top_k=5,
                folder_filter=filters.get('folder_filter'),
                document_type_filter=filters.get('document_type'),
                program_filter=filters.get('program_filter')
            )
        except Exception as e:
            print(f"\r❌ Retrieval error: {e}")
            relevant_docs = []

        # Apply relevance filtering
        filtered = []
        for d in relevant_docs:
            rel = d.get("relevance")
            try: rel = float(rel)
            except (TypeError, ValueError): rel = None
            if rel is None or rel >= min_relevance:
                filtered.append(d)

        # Fallback logic
        if require_context and not filtered:
            if relevant_docs:
                filtered = [relevant_docs[0]]
            else:
                return "I don't have enough information in my Admissions & Aid knowledge base to answer that.", []

        relevant_docs = filtered

        if require_context and not relevant_docs:
            return "I don't have enough information in my Admissions & Aid knowledge base to answer that.", []

        retrieval_time = time.time() - start_time
        if stream:
            print("\r" + " " * 60 + "\r", end="", flush=True)
        print(f"⏱️ Document retrieval: {retrieval_time:.2f}s")
        
        # Continue with existing logic for LLM generation...
        # ... rest of existing process_query method ...

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
    
    def analyze_query_relationship(self, current_query: str, previous_exchanges: list) -> dict:
        """
        Multi-model AI analysis using all available NLP models
        Combines TF-IDF + Word2Vec + Embeddings + Together AI for robust analysis
        """
        if not previous_exchanges:
            return {'needs_history': False, 'confidence': 1.0, 'reason': 'no_history', 'exchanges_needed': 0}
        
        current_lower = current_query.lower().strip()
        last_exchange = previous_exchanges[-1]
        
        # Level 1: Explicit references (high confidence, fast)
        if any(ref in current_lower for ref in ['that', 'this', 'it', 'previous', 'you said', 'these']):
            return {'needs_history': True, 'confidence': 0.95, 'reason': 'explicit_reference', 'exchanges_needed': 1}
        
        print(f"🧠 MULTI-MODEL ANALYSIS:")
        print(f"   Analyzing: '{current_query[:50]}...' vs '{last_exchange['query'][:50]}...'")
        
        # Level 2: Semantic similarity using existing NLP models
        try:
            # Method A: Use Chroma embeddings (if available)
            if self.use_chroma:
                current_emb = _embed_text(current_query)
                previous_emb = _embed_text(last_exchange['query'])
                chroma_similarity = cosine_similarity(
                    current_emb.reshape(1, -1), 
                    previous_emb.reshape(1, -1)
                )[0][0]
                print(f"   Chroma similarity: {chroma_similarity:.3f}")
            else:
                chroma_similarity = None
            
            # Method B: Use TF-IDF + Word2Vec hybrid
            if self.tfidf_vectorizer is not None:
                current_vector = self._vectorize_query(current_query)
                previous_vector = self._vectorize_query(last_exchange['query'])
                hybrid_similarity = cosine_similarity(
                    current_vector.reshape(1, -1),
                    previous_vector.reshape(1, -1)
                )[0][0]
                print(f"   Hybrid TF-IDF+W2V similarity: {hybrid_similarity:.3f}")
            else:
                hybrid_similarity = None
            
            # Combine similarity scores
            similarities = [s for s in [chroma_similarity, hybrid_similarity] if s is not None]
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                max_similarity = max(similarities)
                
                print(f"   Average similarity: {avg_similarity:.3f}")
                print(f"   Max similarity: {max_similarity:.3f}")
                
                # Use the more conservative (lower) score for decisions
                final_similarity = avg_similarity
                
                if final_similarity > 0.6:
                    return {'needs_history': True, 'confidence': 0.8, 'reason': f'high_ai_similarity_{final_similarity:.3f}', 'exchanges_needed': 1}
                elif final_similarity > 0.4:
                    return {'needs_history': True, 'confidence': 0.6, 'reason': f'medium_ai_similarity_{final_similarity:.3f}', 'exchanges_needed': 1}
                elif final_similarity > 0.25:
                    return {'needs_history': True, 'confidence': 0.4, 'reason': f'low_ai_similarity_{final_similarity:.3f}', 'exchanges_needed': 1}
                else:
                    return {'needs_history': False, 'confidence': 0.85, 'reason': f'no_ai_similarity_{final_similarity:.3f}', 'exchanges_needed': 0}
            
        except Exception as e:
            print(f"   🚨 AI similarity analysis failed: {e}")
        
        # Fallback: Conservative approach
        return {'needs_history': False, 'confidence': 0.7, 'reason': 'conservative_fallback', 'exchanges_needed': 0}

    def build_smart_history_context(self, query: str, available_tokens: int) -> str:
        """Build history context based on intelligent analysis"""
        if not self.dialogue_history or available_tokens < 100:
            return ""
        
        # Analyze relationship
        analysis = self.analyze_query_relationship(query, self.dialogue_history)
        
        print(f"🧠 CONTEXT ANALYSIS:")
        print(f"   Needs history: {analysis['needs_history']}")
        print(f"   Confidence: {analysis['confidence']:.2f}")
        print(f"   Reason: {analysis['reason']}")
        print(f"   Exchanges needed: {analysis['exchanges_needed']}")
        
        if not analysis['needs_history']:
            return ""
        
        # Build appropriate history based on confidence and reason
        history_context = ""
        
        if analysis['confidence'] >= 0.8:
            # High confidence - include detailed recent context
            history_context = "Previous conversation:\n"
            exchanges_to_include = min(analysis['exchanges_needed'] + 1, len(self.dialogue_history))
            
            for exchange in reversed(self.dialogue_history[-exchanges_to_include:]):
                entry = f"User: {exchange['query'][:50]}{'...' if len(exchange['query']) > 50 else ''}\nBot: {exchange['response'][:60]}{'...' if len(exchange['response']) > 60 else ''}\n"
                
                if len((history_context + entry).split()) <= available_tokens:
                    history_context += entry
                else:
                    break
                    
        elif analysis['confidence'] >= 0.5:
            # Medium confidence - include condensed recent context
            last = self.dialogue_history[-1]
            history_context = f"Context: {last['query'][:30]}... -> {last['response'][:35]}...\n"
            
        elif analysis['confidence'] >= 0.3:
            # Low confidence - include minimal context
            last = self.dialogue_history[-1]
            history_context = f"Previous: {last['response'][:25]}...\n"
        
        # Safety check
        if len(history_context.split()) > available_tokens:
            return ""
        
        return history_context

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

        # Relax the context filter so valid hits aren't discarded
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
                return "I don't have enough information in my Admissions & Aid knowledge base to answer that.", []

        relevant_docs = filtered

        if require_context and not relevant_docs:
            return "I don't have enough information in my Admissions & Aid knowledge base to answer that.", []

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
            # Calculate available token budget for history
            base_prompt = f"Context information:\n{doc_context}\nQuestion: {query}\nInstructions: You must answer strictly and only using the context above.\nAnswer:"
            base_tokens = len(base_prompt.split())  # This is ~187 tokens
            available_for_history = 3700 - base_tokens  # 3700 - 187 = 3513 tokens
            
            # Use smart history building
            history_context = self.build_smart_history_context(query, available_for_history)
        else:
            history_context = ""

        # Build prompt optimized for complete responses with advanced RAG instructions
        prompt = f"""<|system|>
You are an admissions assistant. Answer questions using ONLY the provided context.

RULES:
- Be direct and concise
- Use simple formatting
- No introductory phrases like "Based on the provided documentation"
- No closing phrases like "I hope this helps"
- Start directly with the answer
- Use numbered lists for steps
- Use bullet points for items
- Bold important terms only when necessary
</|system|>

<|context|>
{doc_context}
</|context|>

<|user|>
{query}
</|user|>

<|assistant|>
"""

        # Generate response using Together AI
        if stream:
            response = stream_response(prompt, max_tokens=max_tokens)
        else:
            print("\n�� Response: ", end="")
            response = generate_response(prompt, max_tokens=max_tokens)
            print(response)

        # Add to history
        self.add_to_history(query, response)

        total_time = time.time() - start_time
        print(f"⏱️ Total processing time: {total_time:.2f}s")

        return response, relevant_docs

    def process_query_stream(self, query: str, correct_spelling: bool = True, max_tokens: int = 3000,
                        use_history: bool = True, require_context: bool = True, 
                        min_relevance: float = 0.1) -> Generator[Dict, None, None]:
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

            # Add debug info before filtering to see what we're getting from ChromaDB
            print(f"🔍 RAW RETRIEVAL DEBUG:")
            print(f"📊 Query: '{query}'")
            print(f"�� Raw retrieved docs: {len(relevant_docs)}")
            for i, doc in enumerate(relevant_docs):
                print(f"   Raw Doc {i+1}: {doc['id']} (Relevance: {doc.get('relevance', 'N/A'):.3f})")
            print(f"📏 Min relevance threshold: {min_relevance}")
            print("-" * 50)

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
                    print(f"✅ PASSED filter: {d['id']} (Relevance: {rel:.3f})")
                else:
                    print(f"❌ FILTERED OUT: {d['id']} (Relevance: {rel:.3f} < {min_relevance})")

            print(f"📄 Docs after filtering: {len(filtered)}")
            print("-" * 50)

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
                f"Source: {doc.get('id','')}\n{doc['content']}"
                for doc in relevant_docs[:3]
            ])

            # History context (same as process_query)
            history_context = ""
            if use_history and self.dialogue_history:
                # Calculate available token budget for history
                base_prompt = f"Context information:\n{doc_context}\nQuestion: {query}\nInstructions: You must answer strictly and only using the context above.\nAnswer:"
                base_tokens = len(base_prompt.split())  # This is ~187 tokens
                available_for_history = 3700 - base_tokens  # 3700 - 187 = 3513 tokens
                
                # Use smart history building
                history_context = self.build_smart_history_context(query, available_for_history)
            else:
                history_context = ""

            # Build prompt optimized for complete responses
            prompt = f"""<|system|>
You are AdmissionsRAG Assistant, a grounded, retrieval-augmented chatbot for first-year admissions. 

CORE INSTRUCTIONS:
- Use ONLY information from the provided ChromaDB context below
- NEVER use external knowledge, web results, or model priors
- NEVER fabricate facts, dates, fees, URLs, names, or policies
- If relevant data is not found in retrieved chunks, use the grounded fallback response
- Cite document titles or IDs from ChromaDB when appropriate (no external links)

COMPLETENESS REQUIREMENT:
- PROVIDE COMPLETE AND COMPREHENSIVE answers using ALL relevant information from the context
- Include ALL steps, requirements, documents, fees, and details mentioned in the context
- Do NOT truncate, summarize, or skip any important information from the retrieved context
- If the context contains multiple sections, include ALL of them in your response

FORMATTING RULES:
- Keep paragraphs short and scannable
- Use clear headings when helpful (e.g., "Enrollment Process", "Required Documents", "Additional Fees", "Curricullum for X Program", " X Year - X Semester Courses")
- When listing items, put each list number on its own line:
  1. First item
  2. Second item
  3. Third item
- Use tables only if the retrieved data is tabular (fees, dates, score bands)
- Bold sparingly for emphasis on critical terms (dates, fees, must-have documents)

TONE & ATTITUDE:
- Friendly, direct, and student-first
- Confident when data is available; transparent when it isn't
- No fluff, no hype, no speculation

SCOPE CONTROL:
- Answer only admissions-related questions using ChromaDB content
- For anything outside admissions scope, decline and redirect to supported topics

FALLBACK RESPONSE:
If retrieval returns no relevant results or confidence is low, respond with:
"I don't have that information in my admissions knowledge base. Here are some ways I can help you:
1. Specify your campus or program of interest
2. Ask about application deadlines or requirements
3. Inquire about specific intake terms"

SAFETY & INTEGRITY:
- Do not provide legal, medical, or immigration advice beyond what's in ChromaDB
- If documents conflict, present the conflict and suggest confirming with admissions office
- Cite sources as: [doc_title or doc_id, section/page if available]
</|system|>

<|context|>
{doc_context}
</|context|>

{history_context}

<|user|>
{query}
</|user|>

<|assistant|>
"""

            print(f"🔍 DEBUG INFO:")
            print(f"📊 Query: '{query}'")
            print(f"📄 Retrieved docs: {len(relevant_docs)}")
            for i, doc in enumerate(relevant_docs):
                print(f"   Doc {i+1}: {doc['id']} (Relevance: {doc.get('relevance', 'N/A'):.3f})")
            print(f"�� History entries: {len(self.dialogue_history)}")
            print(f"📏 Doc context: {len(doc_context)} chars (~{len(doc_context.split())} tokens)")
            print(f"�� History context: {len(history_context)} chars (~{len(history_context.split())} tokens)")
            print(f"�� Total prompt: {len(prompt)} chars (~{len(prompt.split())} tokens)")
            print(f"📏 Estimated tokens left for response: {4096 - len(prompt.split())}")
            print("=" * 70)
            
            # Stream from Together AI
            full_response = ""
            
            # Create streaming config optimized for complete responses
            stream_config = TOGETHER_CONFIG.copy()
            stream_config.update({
                "stream": True,
                "max_tokens": max_tokens,
                "temperature": 0.1,  # Much more deterministic
                "top_p": 0.8,        # More focused
                "top_k": 20,         # More focused
                "repetition_penalty": 1.1,
                "stop": ["</|assistant|>", "<|user|>", "<|system|>", "USER QUESTION:", "CONTEXT:", "<|context|>", "I hope this helps", "Based on the provided"]
            })
            
            try:
                # Generate streaming response from Together AI
                for chunk in llm(prompt, echo=False, **stream_config):
                    if isinstance(chunk, dict):
                        # Handle Together AI response format
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
                yield {"error": f"Together AI generation error: {e}"}
                
        except Exception as e:
            yield {"error": f"Processing error: {e}"}

    def _discover_programs_from_data(self):
        """Automatically discover programs from existing filenames and metadata"""
        try:
            from .models import DocumentMetadata
            
            # Get all document filenames from your database
            documents = DocumentMetadata.objects.filter(
                synced_to_chroma=True
            ).values('filename', 'document_id', 'keywords', 'folder__name')
            
            programs = set()
            filename_patterns = {}
            
            for doc in documents:
                filename = doc['filename'].lower()
                folder_name = doc.get('folder__name', '').lower()
                
                # Focus on subject/curriculum documents
                if 'subject' in folder_name or 'curriculum' in filename or 'program' in filename:
                    # Extract program names from filenames using common patterns
                    extracted_programs = self._extract_programs_from_filename(filename)
                    programs.update(extracted_programs)
                    
                    # Build reverse mapping: program -> filename patterns
                    for program in extracted_programs:
                        if program not in filename_patterns:
                            filename_patterns[program] = set()
                        
                        # Extract keywords from filename
                        filename_keywords = self._extract_keywords_from_filename(filename)
                        filename_patterns[program].update(filename_keywords)
            
            # Cache the discovered programs
            self._program_cache = {
                'programs': list(programs),
                'patterns': {k: list(v) for k, v in filename_patterns.items()},
                'last_updated': time.time()
            }
            
            print(f"🔍 Auto-discovered {len(programs)} programs from filenames")
            print(f"📋 Programs: {', '.join(sorted(programs)[:10])}{'...' if len(programs) > 10 else ''}")
            
        except Exception as e:
            print(f"⚠️ Could not auto-discover programs: {e}")
            # Fallback to basic patterns
            self._setup_fallback_patterns()

    def _extract_programs_from_filename(self, filename: str) -> set:
        """Extract program names from filename using smart patterns"""
        programs = set()
        
        # Common filename patterns
        filename_clean = filename.replace('_', ' ').replace('-', ' ').replace('.pdf', '')
        
        # Split into words and look for program indicators
        words = filename_clean.split()
        
        # Look for multi-word programs
        for i in range(len(words)):
            for j in range(i + 1, min(i + 4, len(words) + 1)):  # Check 1-3 word combinations
                potential_program = ' '.join(words[i:j])
                
                # Filter out common non-program words
                if self._is_likely_program_name(potential_program):
                    programs.add(potential_program)
        
        return programs

    def _is_likely_program_name(self, text: str) -> bool:
        """Determine if text is likely a program name"""
        text_lower = text.lower()
        
        # Skip common non-program words
        skip_words = {
            'curriculum', 'program', 'course', 'courses', 'subject', 'subjects',
            'undergraduate', 'graduate', 'senior', 'high', 'school', 'college',
            'bachelor', 'master', 'phd', 'degree', 'diploma', 'certificate',
            'first', 'second', 'third', 'fourth', 'year', 'semester'
        }
        
        if text_lower in skip_words:
            return False
        
        # Must be at least 2 characters
        if len(text) < 2:
            return False
        
        # Should contain letters
        if not any(c.isalpha() for c in text):
            return False
        
        # Common program name patterns
        program_indicators = [
            'science', 'studies', 'engineering', 'technology', 'management',
            'administration', 'education', 'arts', 'business', 'health',
            'nursing', 'medicine', 'psychology', 'mathematics', 'biology',
            'chemistry', 'physics', 'economics', 'finance', 'accounting'
        ]
        
        # If it contains program indicators, it's likely a program
        if any(indicator in text_lower for indicator in program_indicators):
            return True
        
        # Check for acronyms (BS, MS, etc. followed by letters)
        if len(text) <= 6 and any(c.isupper() for c in text):
            return True
        
        return False

    def _extract_keywords_from_filename(self, filename: str) -> set:
        """Extract searchable keywords from filename"""
        keywords = set()
        
        # Clean filename
        clean = filename.replace('_', ' ').replace('-', ' ').replace('.pdf', '').lower()
        
        # Add individual words
        words = clean.split()
        keywords.update(words)
        
        # Add common abbreviations
        abbreviations = {
            'computer science': ['cs', 'compsci', 'computing'],
            'information technology': ['it', 'infotech'],
            'business administration': ['ba', 'business', 'admin'],
            'engineering': ['engr', 'eng'],
            'mathematics': ['math', 'maths'],
            'bachelor of science': ['bs', 'bsc'],
            'master of science': ['ms', 'msc']
        }
        
        for full_name, abbrevs in abbreviations.items():
            if full_name in clean:
                keywords.update(abbrevs)
        
        return keywords

    def _setup_fallback_patterns(self):
        """Setup basic fallback patterns if auto-discovery fails"""
        self._program_cache = {
            'programs': [
                'computer science', 'business administration', 'engineering',
                'nursing', 'education', 'psychology', 'biology', 'mathematics'
            ],
            'patterns': {
                'computer science': ['cs', 'computing', 'software', 'programming'],
                'business administration': ['business', 'management', 'admin'],
                'engineering': ['engr', 'eng', 'engineering'],
                'nursing': ['nursing', 'health', 'medical'],
                'education': ['education', 'teaching', 'pedagogy'],
                'psychology': ['psych', 'psychology', 'behavioral'],
                'biology': ['bio', 'biology', 'life science'],
                'mathematics': ['math', 'mathematics', 'statistics']
            },
            'last_updated': time.time()
        }

    def _detect_program_from_query_dynamic(self, query_lower: str) -> str:
        """Dynamically detect program from query using discovered patterns"""
        
        # Refresh cache if it's old (once per hour)
        if time.time() - self._program_cache.get('last_updated', 0) > 3600:
            self._discover_programs_from_data()
        
        best_match = None
        best_score = 0
        
        patterns = self._program_cache.get('patterns', {})
        
        for program, keywords in patterns.items():
            score = 0
            
            # Check direct program name match
            if program in query_lower:
                score += 10
            
            # Check keyword matches
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1
            
            # Update best match
            if score > best_score:
                best_score = score
                best_match = program
        
        return best_match if best_score > 0 else None

    def _get_dynamic_filename_patterns(self, program_hint: str) -> List[str]:
        """Get filename patterns dynamically from discovered data"""
        if not program_hint:
            return []
        
        patterns = self._program_cache.get('patterns', {})
        
        # Direct lookup
        if program_hint in patterns:
            return patterns[program_hint]
        
        # Fuzzy matching for partial matches
        for program, keywords in patterns.items():
            if program_hint in program or any(program_hint in keyword for keyword in keywords):
                return keywords
        
        # Fallback to the hint itself
        return [program_hint]

    def _detect_smart_folder_and_filename_dynamic(self, query_lower: str, document_type: str) -> Dict[str, str]:
        """Dynamic folder and filename detection"""
        
        # Detect program using discovered patterns
        detected_program = self._detect_program_from_query_dynamic(query_lower)
        
        # Folder mapping (this stays static)
        folder_mapping = {
            'academic': 'Subjects',  # All curricula go here
            'admission': 'Admissions',
            'enrollment': 'Enrollment',
            'scholarship': 'Financial Aid',
            'fees': 'Fees and Payments',
            'contact': 'Contact Information'
        }
        
        folder_filter = folder_mapping.get(document_type)
        
        return {
            'folder_filter': folder_filter,
            'program_hint': detected_program,
            'document_type': document_type
        }

    def _detect_document_type_with_confidence(self, query_lower: str) -> Tuple[str, float]:
        """Detect document type with confidence score"""
        type_indicators = {
            'admission': (['admission', 'apply', 'application', 'requirement', 'entrance', 'qualify'], 0.9),
            'enrollment': (['enroll', 'registration', 'register', 'sign up'], 0.9),
            'scholarship': (['scholarship', 'financial aid', 'grant', 'funding'], 0.95),
            'academic': (['program', 'course', 'curriculum', 'degree', 'major', 'subject'], 0.8),
            'fees': (['fee', 'cost', 'payment', 'tuition', 'price'], 0.9),
            'contact': (['contact', 'phone', 'email', 'office', 'address'], 0.95)
        }
        
        best_type = None
        best_confidence = 0
        
        for doc_type, (keywords, base_confidence) in type_indicators.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            if matches > 0:
                confidence = base_confidence * (matches / len(keywords))
                if confidence > best_confidence:
                    best_type = doc_type
                    best_confidence = confidence
        
        return best_type, best_confidence

    def _detect_program_with_confidence(self, query_lower: str) -> Tuple[str, float]:
        """Detect program level with confidence"""
        program_indicators = {
            'undergraduate': (['undergraduate', 'bachelor', 'college', 'bscs', 'bsit', 'bsba'], 0.9),
            'graduate': (['graduate', 'master', 'phd', 'doctoral', 'mba', 'ms'], 0.9),
            'senior_high': (['senior high', 'shs', 'grade 11', 'grade 12', 'strand'], 0.95)
        }
        
        for program, (keywords, confidence) in program_indicators.items():
            if any(keyword in query_lower for keyword in keywords):
                return program, confidence
        
        return None, 0.0

    def _retrieve_with_filename_intelligence_dynamic(self, query: str, intent: Dict, top_k: int) -> List[Dict]:
        """Dynamic filename-based retrieval"""
        
        # Get base results from the correct folder
        base_results = self._retrieve_from_chroma(
            query,
            top_k=top_k * 3,  # Get more results for better filtering
            folder_filter=intent.get('folder_filter'),
            document_type_filter=intent.get('document_type'),
            program_filter=intent.get('program_filter')
        )
        
        # Apply dynamic filename filtering
        program_hint = intent.get('program_hint')
        if program_hint and base_results:
            filename_filtered = []
            
            # Get dynamic patterns for this program
            filename_patterns = self._get_dynamic_filename_patterns(program_hint)
            
            for doc in base_results:
                filename = doc.get('filename', '').lower()
                doc_id = doc.get('id', '').lower()
                
                # Calculate filename relevance
                filename_score = 0
                matched_patterns = []
                
                for pattern in filename_patterns:
                    if pattern in filename or pattern in doc_id:
                        filename_score += 1
                        matched_patterns.append(pattern)
                
                if filename_score > 0:
                    doc_copy = doc.copy()
                    doc_copy['relevance'] = doc_copy.get('relevance', 0) + (filename_score * 0.1)
                    doc_copy['filename_matches'] = filename_score
                    doc_copy['matched_patterns'] = matched_patterns
                    filename_filtered.append(doc_copy)
            
            if filename_filtered:
                filename_filtered.sort(key=lambda x: x['relevance'], reverse=True)
                return filename_filtered[:top_k]
        
        return base_results[:top_k]

    def _retrieve_with_metadata_priority(self, query: str, intent: Dict, top_k: int) -> List[Dict]:
        """Retrieve with strong metadata filtering"""
        return self._retrieve_from_chroma(
            query, 
            top_k=top_k,
            folder_filter=None,  # Be flexible on folder
            document_type_filter=intent.get('document_type'),
            program_filter=intent.get('program_filter')
        )

    def _retrieve_with_folder_priority(self, query: str, intent: Dict, top_k: int) -> List[Dict]:
        """Retrieve with folder intelligence"""
        return self._retrieve_from_chroma(
            query,
            top_k=top_k, 
            folder_filter=intent.get('folder_filter'),
            document_type_filter=None,  # Be flexible on type
            program_filter=intent.get('program_filter')
        )

    def _retrieve_semantic_only(self, query: str, top_k: int) -> List[Dict]:
        """Pure semantic retrieval without filters"""
        return self._retrieve_from_chroma(
            query,
            top_k=top_k,
            folder_filter=None,
            document_type_filter=None, 
            program_filter=None
        )

    def _merge_and_rank_candidates(self, all_candidates: List[Tuple], query: str, top_k: int) -> List[Dict]:
        """Intelligent merging and ranking of candidates from different strategies"""
        
        # Deduplicate by document ID
        seen_ids = set()
        unique_candidates = []
        
        for doc, strategy, boosted_score in all_candidates:
            doc_id = doc.get('id', '')
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                
                # Calculate final score
                final_score = self._calculate_hybrid_score(doc, strategy, boosted_score, query)
                
                doc_enhanced = doc.copy()
                doc_enhanced.update({
                    'hybrid_score': final_score,
                    'retrieval_strategy': strategy,
                    'original_relevance': doc.get('relevance', 0)
                })
                unique_candidates.append(doc_enhanced)
        
        # Sort by hybrid score and return top-k
        unique_candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return unique_candidates[:top_k]

    def _calculate_hybrid_score(self, doc: Dict, strategy: str, boosted_score: float, query: str) -> float:
        """Enhanced scoring with filename intelligence"""
        
        base_score = boosted_score
        query_terms = set(query.lower().split())
        
        # Strategy bonuses (filename gets highest priority)
        strategy_bonus = {
            'filename': 0.20,   # Highest - filename matches are very relevant
            'metadata': 0.15,   # High - precise metadata matches
            'folder': 0.10,     # Medium - folder organization
            'semantic': 0.05    # Base - semantic similarity
        }.get(strategy, 0)
        
        # Filename quality bonus
        filename = doc.get('filename', '').lower()
        filename_bonus = 0
        
        # Check for comprehensive filename patterns
        filename_terms = set(filename.replace('_', ' ').replace('-', ' ').split())
        
        # Filename term coverage
        coverage = len(query_terms.intersection(filename_terms)) / len(query_terms) if query_terms else 0
        filename_bonus += coverage * 0.15
        
        # Program specificity bonus
        if any(prog in filename for prog in ['undergraduate', 'graduate', 'senior_high', 'curriculum']):
            filename_bonus += 0.05
        
        # Content quality signals
        content = doc.get('content', '')
        content_bonus = 0
        
        # Length preference for curricula
        content_length = len(content)
        if strategy == 'filename':  # Curricula should be comprehensive
            if content_length > 2000:  # Substantial curriculum content
                content_bonus += 0.10
        else:
            if 500 <= content_length <= 3000:  # Standard documents
                content_bonus += 0.05
        
        # Query term coverage in content
        content_terms = set(content.lower().split())
        content_coverage = len(query_terms.intersection(content_terms)) / len(query_terms) if query_terms else 0
        content_bonus += content_coverage * 0.08
        
        final_score = base_score + strategy_bonus + filename_bonus + content_bonus
        
        return min(final_score, 1.0)  # Cap at 1.0

def test_fast_hybrid_chatbot_together():
    """Test the fast hybrid chatbot with Together AI"""
    chatbot = FastHybridChatbotTogether(use_chroma=True)
    
    print("\n⚡🔍 FAST HYBRID CHATBOT WITH TOGETHER AI ⚡🔍")
    print("Type 'exit' to quit\n")

    max_tokens = 500  # Higher default for complete responses
