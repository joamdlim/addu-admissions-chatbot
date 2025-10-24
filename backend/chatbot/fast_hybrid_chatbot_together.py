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
    print("✓ Gensim Word2Vec available")
except ImportError:
    WORD2VEC_AVAILABLE = False
    print("Warning: Gensim Word2Vec not available, using placeholder")

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
from chatbot.topics import (
    CONVERSATION_STATES, get_button_configs,
    get_topic_keywords, find_matching_topics, get_topic_info,
    get_topic_retrieval_strategy, get_retrieval_strategy_config
)

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
        
        # Session state for guided conversation
        self.session_state = {
            'current_topic': None,
            'conversation_state': CONVERSATION_STATES['TOPIC_SELECTION'],
            'session_id': None
        }
        
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
    
    def retrieve_documents_by_topic_keywords_simple(self, query: str, topic_id: str, top_k: int = 2) -> List[Dict]:
        """
        Simplified topic-filtered retrieval that avoids complex ChromaDB queries.
        Uses keyword filtering first, then basic semantic search.
        """
        import re
        
        print(f"🎯 Simple topic-filtered retrieval for: '{query}' (topic: {topic_id})")
        
        try:
            collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)
            
            # Get topic keywords
            topic_keywords = get_topic_keywords(topic_id)
            if not topic_keywords:
                print(f"⚠️ No keywords found for topic: {topic_id}")
                return []
            
            print(f"📝 Topic keywords: {topic_keywords}")
            
            # Get ALL documents to filter by keywords
            all_docs = collection.get(
                where={"source": "pdf_scrape"},
                include=["documents", "metadatas"]
            )
            
            all_ids = all_docs.get('ids', [])
            all_contents = all_docs.get('documents', [])
            all_metadatas = all_docs.get('metadatas', [])
            
            print(f"📚 Searching through {len(all_ids)} documents...")
            
            # Filter documents by topic keywords
            topic_filtered_results = []
            for i, (doc_id, content, metadata) in enumerate(zip(all_ids, all_contents, all_metadatas)):
                doc_keywords = metadata.get('keywords', '').lower()
                filename = metadata.get('filename', '').lower()
                
                # Check if document keywords match any topic keywords
                keyword_matches = 0
                matched_keywords = []
                
                for topic_keyword in topic_keywords:
                    topic_keyword_lower = topic_keyword.lower()
                    # Use word boundaries to avoid substring matches
                    pattern = r'\b' + re.escape(topic_keyword_lower) + r'\b'
                    
                    if re.search(pattern, doc_keywords) or re.search(pattern, filename):
                        keyword_matches += 1
                        matched_keywords.append(topic_keyword)
                
                # Only include documents that match at least one topic keyword
                if keyword_matches > 0:
                    topic_relevance = keyword_matches / len(topic_keywords)
                    
                    topic_filtered_results.append({
                        'id': doc_id,
                        'content': content,
                        'relevance': topic_relevance,
                        'folder': metadata.get('folder_name', 'Unknown'),
                        'document_type': metadata.get('document_type', 'other'),
                        'target_program': metadata.get('target_program', 'all'),
                        'filename': metadata.get('filename', ''),
                        'retrieval_strategy': 'simple-topic-filtered',
                        'current_topic': topic_id,
                        '_debug': {
                            'topic_score': topic_relevance,
                            'semantic_score': 0.0,
                            'topic_keyword_matches': keyword_matches,
                            'matched_keywords': matched_keywords
                        }
                    })
            
            print(f"🎯 Found {len(topic_filtered_results)} documents matching topic keywords")
            
            if not topic_filtered_results:
                print(f"❌ No documents found for topic: {topic_id}")
                return []
            
            # Sort by topic relevance and return top results
            topic_filtered_results.sort(key=lambda x: x['relevance'], reverse=True)
            
            # Debug output
            print(f"✅ Top {min(top_k, len(topic_filtered_results))} simple topic-filtered results:")
            for i, doc in enumerate(topic_filtered_results[:top_k]):
                debug = doc['_debug']
                print(f"   {i+1}. {doc['filename'][:70]}")
                print(f"       Score: {doc['relevance']:.3f} (topic matches: {debug['topic_keyword_matches']})")
                print(f"       Matched keywords: {debug['matched_keywords']}")
            
            return topic_filtered_results[:top_k]
            
        except Exception as e:
            print(f"❌ Simple topic-filtered retrieval error: {e}")
            import traceback
            traceback.print_exc()
            return []

    # ===== HELPER METHODS FOR SPECIALIZED RETRIEVAL =====
    
    def _detect_student_type(self, query: str) -> str:
        """Detect student type from query for admissions specialization"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['transfer', 'shifter', 'lateral']):
            return 'transfer'
        elif any(term in query_lower for term in ['international', 'foreign', 'overseas']):
            return 'international'
        elif any(term in query_lower for term in ['scholar', 'scholarship', 'financial aid']):
            return 'scholar'
        elif any(term in query_lower for term in ['new student', 'freshman', 'first year', 'incoming']):
            return 'new'
        else:
            return 'general'
    
    def _detect_requirement_type(self, query: str) -> str:
        """Detect requirement type from query"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['documents', 'requirements', 'needed', 'submit']):
            return 'documents'
        elif any(term in query_lower for term in ['process', 'procedure', 'steps', 'how to']):
            return 'process'
        elif any(term in query_lower for term in ['exam', 'test', 'assessment']):
            return 'examination'
        else:
            return 'general'
    
    def _get_program_patterns(self) -> dict:
        """Get comprehensive ADDU programs list with patterns for matching"""
        return {
            # Business and Governance
            'accountancy': ['accountancy', 'bsa', 'bs a', 'accounting'],
            'management accounting': ['management accounting', 'bsma', 'bs ma'],
            'business management': ['business management', 'bsbm', 'bs bm'],
            'entrepreneurship': ['entrepreneurship', 'bs entrep', 'bsentrep', 'entrepreneur'],
            'finance': ['finance', 'bsfin', 'bs fin'],
            'human resource development': ['human resource', 'hrdm', 'bshrdm', 'bs hrdm', 'hr'],
            'marketing': ['marketing', 'bs mktg', 'bsmktg'],
            'public management': ['public management', 'bpm', 'governance'],
            
            # Technology Programs
            'computer science': ['computer science', 'cs', 'bscs', 'bs cs', 'compsci', 'comsci'],
            'information technology': ['information technology', 'it', 'bsit', 'bs it', 'infotech'],
            'information systems': ['information systems', 'bsis', 'bs is'],
            'data science': ['data science', 'bsds', 'bs ds'],
            
            # Science Programs
            'biology': ['biology', 'bsbio', 'bs bio', 'bio'],
            'chemistry': ['chemistry', 'bschem', 'bs chem', 'chem'],
            'mathematics': ['mathematics', 'bsmath', 'bs math', 'math'],
            'environmental science': ['environmental science', 'bsenvisci', 'bs envisci', 'envisci'],
            'social work': ['social work', 'bssocialwork', 'bs social work', 'bssw'],
            
            # Arts Programs
            'anthropology': ['anthropology', 'abanthro', 'ab anthro', 'abanth', 'anthro'],
            'communication': ['communication', 'abc', 'ab c', 'abcomm', 'comm'],
            'development studies': ['development studies', 'abds', 'ab ds'],
            'economics': ['economics', 'abecon', 'ab econ', 'econ'],
            'english language': ['english language', 'abel', 'ab el', 'english'],
            'interdisciplinary studies': ['interdisciplinary', 'abis', 'ab is'],
            'international studies': ['international studies', 'abis', 'ab is'],
            'islamic studies': ['islamic studies', 'abis', 'ab is', 'islamic'],
            'philosophy': ['philosophy', 'abphilo', 'ab philo', 'philo'],
            'political science': ['political science', 'abpolsci', 'ab polsci', 'polsci'],
            'psychology': ['psychology', 'abpsych', 'ab psych', 'psych'],
            'sociology': ['sociology', 'absocio', 'ab socio', 'socio'],
            
            # Education
            'early childhood education': ['early childhood', 'bece', 'ece'],
            'elementary education': ['elementary education', 'beed', 'elem ed'],
            'secondary education': ['secondary education', 'bsed', 'sec ed'],
            
            # Engineering and Architecture
            'aerospace engineering': ['aerospace', 'bsae', 'bs ae', 'aero'],
            'architecture': ['architecture', 'bsarch', 'bs arch', 'archi'],
            'chemical engineering': ['chemical engineering', 'bsche', 'bs che', 'chemeng'],
            'civil engineering': ['civil engineering', 'bsce', 'bs ce', 'civil'],
            'computer engineering': ['computer engineering', 'bscompeng', 'bs comp eng', 'bscpe', 'compeng'],
            'electrical engineering': ['electrical engineering', 'bsee', 'bs ee', 'electrical'],
            'electronics engineering': ['electronics engineering', 'bselectronicseng', 'bs electronics eng', 'electronics'],
            'industrial engineering': ['industrial engineering', 'bsie', 'bs ie', 'industrial'],
            'mechanical engineering': ['mechanical engineering', 'bsme', 'bs me', 'mechanical'],
            'robotics engineering': ['robotics', 'bsre', 'bs re', 'robot'],
            
            # Nursing
            'nursing': ['nursing', 'bsn', 'nurse']
        }

    def _extract_program_info(self, query: str) -> dict:
        """Extract program information from query for programs specialization"""
        query_lower = query.lower()
        
        program_info = {
            'program_name': None,
            'degree_level': None,
            'year_level': None,
            'course_code': None,
            'context_source': None
        }
        
        # Detect program names using helper method with word boundary matching
        program_patterns = self._get_program_patterns()
        
        import re
        for program, patterns in program_patterns.items():
            for pattern in patterns:
                # Use word boundaries for short patterns that could be substrings
                if len(pattern) <= 3:
                    # Use word boundary regex for short patterns like 'it', 'cs', etc.
                    if re.search(r'\b' + re.escape(pattern) + r'\b', query_lower):
                        program_info['program_name'] = program
                        break
                else:
                    # Use simple substring matching for longer patterns
                    if pattern in query_lower:
                        program_info['program_name'] = program
                        break
            if program_info['program_name']:
                break
        
        # Detect degree level
        if any(term in query_lower for term in ['undergraduate', 'bachelor', 'bs', 'ba']):
            program_info['degree_level'] = 'undergraduate'
        elif any(term in query_lower for term in ['graduate', 'master', 'ms', 'ma']):
            program_info['degree_level'] = 'graduate'
        elif any(term in query_lower for term in ['senior high', 'shs']):
            program_info['degree_level'] = 'senior_high'
        
        # Detect year level
        year_patterns = {
            'first': ['first year', '1st year', 'freshman'],
            'second': ['second year', '2nd year', 'sophomore'],
            'third': ['third year', '3rd year', 'junior'],
            'fourth': ['fourth year', '4th year', 'senior']
        }
        
        for year, patterns in year_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                program_info['year_level'] = year
                break
        
        return program_info
    
    def _extract_program_info_with_history(self, query: str) -> dict:
        """Extract program information with conversation history awareness for follow-up questions"""
        query_lower = query.lower().strip()
        
        # CRITICAL: Check for pronouns that might be misinterpreted as program codes BEFORE extraction
        pronoun_indicators = ['its', 'it', 'this', 'that', 'the program', 'the course']
        curriculum_indicators = ['curriculum', 'courses', 'subjects', 'syllabus', 'course outline', 'academic plan']
        general_program_indicators = ['tell me about', 'about it', 'information about', 'details about']
        
        # Check for pronoun context first
        has_pronoun = any(pronoun in query_lower for pronoun in pronoun_indicators)
        has_curriculum_context = any(term in query_lower for term in curriculum_indicators)
        has_general_program_context = any(term in query_lower for term in general_program_indicators)
        
        # If this looks like a pronoun reference, skip normal program extraction and go to history
        if has_pronoun and (has_curriculum_context or has_general_program_context):
            context_type = "curriculum" if has_curriculum_context else "general program"
            print(f"🔍 Detected pronoun reference with {context_type} context: '{query}' - skipping normal extraction")
            
            program_info = {
                'program_name': None,
                'degree_level': None,
                'year_level': None,
                'course_code': None,
                'context_source': None
            }
            
            # PRIORITY 1: Check recent queries first (most reliable)
            if self.dialogue_history:
                for i, exchange in enumerate(reversed(self.dialogue_history[-3:])):
                    prev_query = exchange.get('query', '')
                    prev_program_info = self._extract_program_info(prev_query)
                    if prev_program_info.get('program_name'):
                        program_info['program_name'] = prev_program_info['program_name']
                        program_info['context_source'] = f'pronoun_resolution_query_{i+1}'
                        print(f"✅ Resolved pronoun to program '{program_info['program_name']}' from: '{prev_query[:50]}...'")
                        return program_info
            
            # PRIORITY 2: Check session state
            if hasattr(self, 'session_state'):
                session_program = self.session_state.get('current_program')
                if session_program:
                    program_info['program_name'] = session_program
                    program_info['context_source'] = 'pronoun_resolution_session'
                    print(f"✅ Resolved pronoun to program '{session_program}' from session state")
                    return program_info
        
        # Normal program extraction if not a pronoun reference
        program_info = self._extract_program_info(query)
        
        # If no program found in current query and no pronoun context, check conversation history normally
        if not program_info.get('program_name'):
            print(f"🔍 No program in current query '{query}', checking conversation history...")
            
            # PRIORITY 1: Check recent queries first (most reliable) - only if dialogue history exists
            if self.dialogue_history:
                for i, exchange in enumerate(reversed(self.dialogue_history[-3:])):
                    prev_query = exchange.get('query', '')
                    prev_program_info = self._extract_program_info(prev_query)
                    if prev_program_info.get('program_name'):
                        program_info['program_name'] = prev_program_info['program_name']
                        program_info['context_source'] = f'query_history_{i+1}'
                        print(f"✅ Found program '{program_info['program_name']}' in previous query: '{prev_query[:50]}...'")
                        return program_info  # Return immediately - highest priority
            
            # PRIORITY 2: Check session state (if no query context found)
            if hasattr(self, 'session_state'):
                session_program = self.session_state.get('current_program')
                if session_program:
                    program_info['program_name'] = session_program
                    program_info['context_source'] = 'session_state'
                    print(f"✅ Found program '{session_program}' in session state")
                    return program_info
            
            # PRIORITY 3: Check response context only as last resort - only if dialogue history exists
            if self.dialogue_history:
                for i, exchange in enumerate(reversed(self.dialogue_history[-2:])):  # Only check last 2 responses
                    prev_response = exchange.get('response', '')
                    
                    # Be more selective - only check if response is specifically about curriculum/courses
                    if any(keyword in prev_response.lower() for keyword in ['curriculum', 'courses', 'subjects', 'year']):
                        program_patterns = self._get_program_patterns()
                        for prog_name, patterns in program_patterns.items():
                            if any(pattern in prev_response.lower() for pattern in patterns):
                                program_info['program_name'] = prog_name
                                program_info['context_source'] = f'response_history_{i+1}'
                                print(f"✅ Found program '{prog_name}' in previous response context (fallback)")
                                return program_info
        
        return program_info
    
    def _extract_fee_info(self, query: str) -> dict:
        """Extract fee information from query for fees specialization"""
        import re
        query_lower = query.lower()
        
        fee_info = {
            'fee_type': None,
            'amount_mentioned': False,
            'payment_term': None,
            'payment_method': None,  # NEW: Payment method detection
            'administrative_action': None,  # NEW: Administrative action detection
            'program_level': None,
            'program_name': None
        }
        
        # Detect fee types
        fee_types = {
            'tuition': ['tuition', 'tuition fee'],
            'miscellaneous': ['miscellaneous', 'misc fee', 'other fees'],
            'laboratory': ['laboratory', 'lab fee'],
            'registration': ['registration', 'enrollment fee'],
            'graduation': ['graduation', 'graduation fee'],
            'loa': ['loa', 'leave of absence'],  # NEW
            'withdrawal': ['withdrawal', 'dropping'],  # NEW
            'administrative': ['administrative fee', 'processing fee']  # NEW
        }
        
        for fee_type, patterns in fee_types.items():
            if any(pattern in query_lower for pattern in patterns):
                fee_info['fee_type'] = fee_type
                break
        
        # Check if amount is mentioned
        if re.search(r'\d+', query_lower) or any(term in query_lower for term in ['cost', 'price', 'amount', 'how much']):
            fee_info['amount_mentioned'] = True
        
        # Detect payment terms
        if any(term in query_lower for term in ['installment', 'payment plan', 'schedule']):
            fee_info['payment_term'] = 'installment'
        elif any(term in query_lower for term in ['full payment', 'lump sum']):
            fee_info['payment_term'] = 'full'
        
        # NEW: Detect payment methods
        payment_methods = {
            'online': ['online', 'online payment', 'pay online', 'e-payment', 'digital'],
            'bank': ['bank', 'bank payment', 'bank deposit', 'bank transfer', 'over the counter', 'otc'],
            'cheque': ['cheque', 'check', 'check payment'],
            'cash': ['cash', 'cash payment'],
            'gcash': ['gcash', 'g-cash'],
            'paymaya': ['paymaya', 'pay maya'],
        }
        
        for method, patterns in payment_methods.items():
            if any(pattern in query_lower for pattern in patterns):
                fee_info['payment_method'] = method
                break
        
        # NEW: Detect administrative actions
        if any(term in query_lower for term in ['loa', 'leave of absence']):
            fee_info['administrative_action'] = 'loa'
        elif any(term in query_lower for term in ['withdrawal', 'withdraw', 'dropping', 'drop']):
            fee_info['administrative_action'] = 'withdrawal'
        elif any(term in query_lower for term in ['refund', 'reimbursement']):
            fee_info['administrative_action'] = 'refund'
        
        # Detect program level for fee differentiation
        if any(term in query_lower for term in ['undergraduate', 'bachelor']):
            fee_info['program_level'] = 'undergraduate'
        elif any(term in query_lower for term in ['graduate', 'master']):
            fee_info['program_level'] = 'graduate'
        
        # Extract program name from query
        program_keywords = [
            # Business and Governance
            'bpm', 'bsa', 'bsma', 'bsbm', 'bsentrep', 'bsfin', 'bshrdm', 'bsmktg',
            'public management', 'accountancy', 'management accounting', 'business management', 
            'entrepreneurship', 'finance', 'human resource development management', 'marketing',
            
            # Arts and Sciences - Technology
            'bsit', 'bscs', 'bsis', 'bsds', 'bs it', 'bs cs', 'bs is', 'bs ds',
            'information technology', 'computer science', 'information systems', 'data science',
            
            # Arts and Sciences - Science
            'bsbio', 'bschem', 'bsmath', 'bsenvisci', 'bssocialwork', 'bs bio', 'bs chem', 'bs math', 'bs envisci', 'bs social work',
            'biology', 'chemistry', 'mathematics', 'environmental science', 'social work',
            
            # Arts and Sciences - Arts
            'abanthro', 'abanth', 'abc', 'abcomm', 'abds', 'abecon', 'abel', 'abis', 'abphilo', 'abpolsci', 'abpsych', 'absocio',
            'ab anthro', 'ab c', 'ab ds', 'ab econ', 'ab el', 'ab is', 'ab philo', 'ab polsci', 'ab psych', 'ab socio',
            'anthropology', 'communication', 'development studies', 'economics', 'english language', 
            'interdisciplinary studies', 'international studies', 'islamic studies', 'philosophy', 
            'political science', 'psychology', 'sociology',
            
            # Education
            'bece', 'beed', 'bsed', 'early childhood education', 'elementary education', 'secondary education',
            
            # Engineering and Architecture
            'bsae', 'bsarch', 'bsche', 'bsce', 'bscompeng', 'bscpe', 'bsee', 'bbselectronicseng', 'bsie', 'bsme', 'bsre',
            'bs ae', 'bs arch', 'bs che', 'bs ce', 'bs comp eng', 'bs ee', 'bs electronics eng', 'bs ie', 'bs me', 'bs re',
            'aerospace engineering', 'architecture', 'chemical engineering', 'civil engineering', 
            'computer engineering', 'electrical engineering', 'electronics engineering', 
            'industrial engineering', 'mechanical engineering', 'robotics engineering',
            
            # Nursing
            'bsn', 'nursing'
        ]
        
        for program in program_keywords:
            if program in query_lower:
                fee_info['program_name'] = program
                break
        
        return fee_info
    
    def _calculate_admissions_filename_score(self, filename: str, student_type: str, requirement_type: str) -> float:
        """Calculate filename score for admissions documents"""
        score = 0.0
        
        # Base score for admissions-related filenames
        if any(term in filename for term in ['admission', 'enrollment', 'requirement']):
            score += 0.5
        
        # Bonus for student type match
        if student_type != 'general':
            if student_type in filename:
                score += 0.3
        
        # Bonus for requirement type match
        if requirement_type != 'general':
            if requirement_type in filename or (requirement_type == 'documents' and 'requirement' in filename):
                score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_programs_filename_score(self, filename: str, program_info: dict) -> float:
        """Calculate filename score for programs documents"""
        score = 0.0
        
        # Base score for program-related filenames
        if any(term in filename for term in ['curriculum', 'program', 'course', 'syllabus']):
            score += 0.4
        
        # Bonus for program name match
        if program_info['program_name'] and program_info['program_name'].replace(' ', '') in filename.replace(' ', ''):
            score += 0.4
        
        # Bonus for year level match
        if program_info['year_level']:
            year_patterns = {
                'first': ['1st', 'first', '1'],
                'second': ['2nd', 'second', '2'],
                'third': ['3rd', 'third', '3'],
                'fourth': ['4th', 'fourth', '4']
            }
            patterns = year_patterns.get(program_info['year_level'], [])
            if any(pattern in filename for pattern in patterns):
                score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_fees_filename_score(self, filename: str, fee_info: dict) -> float:
        """Calculate filename score for fees documents"""
        score = 0.0
        
        # Base score for fee-related filenames
        if any(term in filename for term in ['fee', 'tuition', 'payment', 'cost']):
            score += 0.4
        
        # Bonus for specific fee type match
        if fee_info['fee_type'] and fee_info['fee_type'] in filename:
            score += 0.3
        
        # Bonus for program level match
        if fee_info['program_level'] and fee_info['program_level'] in filename:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_keyword_score(self, doc_keywords: str, query_lower: str) -> float:
        """Calculate keyword matching score (shared across specializations)"""
        if not doc_keywords or not query_lower:
            return 0.0
        
        query_terms = set(query_lower.split())
        keyword_terms = set(doc_keywords.split())
        
        if not query_terms:
            return 0.0
        
        matches = len(query_terms.intersection(keyword_terms))
        return matches / len(query_terms)
    
    def _calculate_admissions_content_score(self, content: str, query_lower: str, student_type: str) -> float:
        """Calculate content score for admissions documents"""
        score = 0.0
        query_terms = query_lower.split()
        
        # Basic term matching
        matches = sum(1 for term in query_terms if term in content)
        if query_terms:
            score += (matches / len(query_terms)) * 0.5
        
        # Bonus for student type context
        if student_type != 'general' and student_type in content:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_programs_content_score(self, content: str, query_lower: str, program_info: dict) -> float:
        """Calculate content score for programs documents"""
        score = 0.0
        query_terms = query_lower.split()
        
        # Basic term matching
        matches = sum(1 for term in query_terms if term in content)
        if query_terms:
            score += (matches / len(query_terms)) * 0.4
        
        # Bonus for program name in content
        if program_info['program_name'] and program_info['program_name'] in content:
            score += 0.4
        
        return min(score, 1.0)
    
    def _calculate_fees_content_score(self, content: str, query_lower: str, fee_info: dict) -> float:
        """Calculate content score for fees documents"""
        score = 0.0
        query_terms = query_lower.split()
        
        # Basic term matching
        matches = sum(1 for term in query_terms if term in content)
        if query_terms:
            score += (matches / len(query_terms)) * 0.2  # Reduced to make room for other bonuses
        
        # Bonus for fee type in content
        if fee_info['fee_type'] and fee_info['fee_type'] in content:
            score += 0.15
        
        # High bonus for program name in content
        if fee_info['program_name'] and fee_info['program_name'] in content:
            score += 0.3
        
        # NEW: Bonus for payment method in content
        if fee_info.get('payment_method') and fee_info['payment_method'] in content:
            score += 0.2
        
        # NEW: Bonus for administrative action in content
        if fee_info.get('administrative_action') and fee_info['administrative_action'] in content:
            score += 0.25
        
        # Bonus for amount-related content if amount was mentioned in query
        if fee_info['amount_mentioned'] and any(term in content for term in ['php', 'peso', 'amount', 'cost']):
            score += 0.15
        
        # NEW: Bonus for regulation/policy documents when asking "how to pay"
        if any(term in query_lower for term in ['how to pay', 'payment method', 'payment procedure']):
            if any(term in content for term in ['regulation', 'policy', 'procedure', 'guidelines']):
                score += 0.2
        
        return min(score, 1.0)

    # ===== NLP-INTEGRATED HELPER METHODS =====
    
    def _calculate_nlp_stem_overlap(self, query_stems: set, text: str) -> float:
        """
        Calculate stem overlap between query and text using NLP preprocessing.
        Both query and text go through the same preprocessing pipeline.
        
        Args:
            query_stems: Set of preprocessed query tokens (stems)
            text: Raw text to compare against
            
        Returns:
            Float between 0.0 and 1.0 representing overlap ratio
        """
        if not query_stems or not text:
            return 0.0
        
        # Preprocess text using same NLP pipeline as query
        text_tokens = preprocess_text(text[:5000])  # Limit for performance
        text_stems = set(text_tokens)
        
        # Calculate overlap
        overlap = len(query_stems & text_stems) / len(query_stems)
        return overlap
    
    def _calculate_nlp_keyword_boost(self, query_stems: set, content: str, 
                                     filename: str, keywords: str) -> dict:
        """
        Calculate keyword boost scores using NLP preprocessing.
        Returns detailed breakdown of different overlap scores.
        
        Args:
            query_stems: Set of preprocessed query tokens
            content: Document content
            filename: Document filename
            keywords: Document metadata keywords
            
        Returns:
            Dict with overlap scores and total boost
        """
        # Calculate overlaps for different fields
        content_overlap = self._calculate_nlp_stem_overlap(query_stems, content)
        filename_overlap = self._calculate_nlp_stem_overlap(query_stems, filename)
        keyword_overlap = self._calculate_nlp_stem_overlap(query_stems, keywords)
        
        # Calculate weighted boost
        # Filename is most reliable, then keywords, then content
        total_boost = (
            content_overlap * 0.10 +    # Content: up to 10%
            filename_overlap * 0.15 +    # Filename: up to 15%
            keyword_overlap * 0.05       # Keywords: up to 5%
        )
        
        return {
            'content_overlap': content_overlap,
            'filename_overlap': filename_overlap,
            'keyword_overlap': keyword_overlap,
            'total_boost': total_boost
        }

    # ===== SPECIALIZED TOPIC RETRIEVAL FUNCTIONS =====
    
    def retrieve_admissions_documents(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        NLP-INTEGRATED specialized retrieval for admissions, enrollment, requirements, and documents.
        Uses TF-IDF + Word2Vec semantic search with specialization bonuses.
        """
        import re
        
        print(f"🎓 NLP-Integrated Admissions Retrieval for: '{query}'")
        
        try:
            collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)
            
            # STEP 1: NLP Preprocessing
            processed_tokens = preprocess_text(query)
            query_stems = set(processed_tokens)
            print(f"📝 Preprocessed query tokens: {processed_tokens[:10]}..." if len(processed_tokens) > 10 else f"📝 Preprocessed query tokens: {processed_tokens}")
            
            # STEP 2: Generate hybrid embedding (TF-IDF + Word2Vec)
            q_emb = _embed_text(query)
            print(f"🧮 Generated hybrid embedding, dimension: {len(q_emb)}")
            
            # STEP 3: Detect specialization context
            student_type = self._detect_student_type(query)
            requirement_type = self._detect_requirement_type(query)
            print(f"📝 Detected: student_type={student_type}, requirement_type={requirement_type}")
            
            # STEP 4: Get document type filters
            strategy_config = get_retrieval_strategy_config('admissions_specialized')
            document_types = strategy_config.get('document_types', ['admission', 'enrollment', 'scholarship'])
            print(f"📋 Document type filters: {document_types}")
            
            # STEP 5: Semantic search with document type filtering
            where_clause = {
                "$and": [
                    {"source": "pdf_scrape"},
                    {"$or": [{"document_type": doc_type} for doc_type in document_types]}
                ]
            }
            
            semantic_results = collection.query(
                query_embeddings=[q_emb],
                n_results=min(top_k * 15, 60),  # Get more candidates for re-ranking
                include=["documents", "distances", "metadatas"],
                where=where_clause
            )
            
            ids = semantic_results.get("ids", [[]])[0]
            docs = semantic_results.get("documents", [[]])[0]
            metadatas = semantic_results.get("metadatas", [[]])[0]
            distances = semantic_results.get("distances", [[]])[0]
            
            print(f"📊 Retrieved {len(ids)} semantic candidates with document_type filtering")
            
            if not ids:
                print("⚠️ No documents found, trying without strict document_type filter")
                # Fallback: search without strict document_type filter
                semantic_results = collection.query(
                    query_embeddings=[q_emb],
                    n_results=min(top_k * 15, 60),
                    include=["documents", "distances", "metadatas"],
                    where={"source": "pdf_scrape"}
                )
                ids = semantic_results.get("ids", [[]])[0]
                docs = semantic_results.get("documents", [[]])[0]
                metadatas = semantic_results.get("metadatas", [[]])[0]
                distances = semantic_results.get("distances", [[]])[0]
                print(f"📊 Retrieved {len(ids)} candidates without document_type filter")
            
            if not ids:
                return []
            
            # STEP 6: Calculate NLP-integrated scores with specialization bonuses
            scored_results = []
            
            for i, (doc_id, content, metadata, distance) in enumerate(zip(ids, docs, metadatas, distances)):
                # A. Semantic score (from TF-IDF + Word2Vec embeddings)
                semantic_score = float(1.0 / (1.0 + distance))
                
                # B. NLP keyword boost
                filename = metadata.get('filename', '')
                doc_keywords = metadata.get('keywords', '')
                boost_data = self._calculate_nlp_keyword_boost(query_stems, content, filename, doc_keywords)
                
                # C. Specialization bonuses
                student_type_bonus = 0.0
                requirement_type_bonus = 0.0
                
                # Bonus for student type match
                if student_type != 'general':
                    content_lower = content.lower()
                    filename_lower = filename.lower()
                    if student_type in content_lower or student_type in filename_lower:
                        student_type_bonus = 0.05
                
                # Bonus for requirement type match
                if requirement_type != 'general':
                    content_lower = content.lower()
                    filename_lower = filename.lower()
                    requirement_terms = {
                        'documents': ['document', 'requirement', 'needed'],
                        'process': ['process', 'procedure', 'step', 'how to'],
                        'examination': ['exam', 'test', 'assessment']
                    }
                    if requirement_type in requirement_terms:
                        if any(term in content_lower for term in requirement_terms[requirement_type]):
                            requirement_type_bonus = 0.05
                
                # D. FINAL SCORE: Semantic (60%) + Keyword Boost (30%) + Specialization (10%)
                final_score = (
                    semantic_score * 0.6 +
                    boost_data['total_boost'] +
                    student_type_bonus +
                    requirement_type_bonus
                )
                
                scored_results.append({
                    'id': doc_id,
                    'content': content,
                    'relevance': final_score,
                    'folder': metadata.get('folder_name', 'Unknown'),
                    'document_type': metadata.get('document_type', 'other'),
                    'target_program': metadata.get('target_program', 'all'),
                    'filename': metadata.get('filename', ''),
                    'retrieval_strategy': 'admissions_nlp_integrated',
                    'current_topic': 'admissions_enrollment',
                    '_debug': {
                        'semantic_score': semantic_score,
                        'semantic_distance': float(distance),
                        'keyword_boost': boost_data['total_boost'],
                        'content_overlap': boost_data['content_overlap'],
                        'filename_overlap': boost_data['filename_overlap'],
                        'keyword_overlap': boost_data['keyword_overlap'],
                        'student_type_bonus': student_type_bonus,
                        'requirement_type_bonus': requirement_type_bonus,
                        'student_type': student_type,
                        'requirement_type': requirement_type
                    }
                })
            
            # Sort and return top results
            scored_results.sort(key=lambda x: x['relevance'], reverse=True)
            
            print(f"✅ Top {min(top_k, len(scored_results))} NLP-integrated admissions results:")
            for i, doc in enumerate(scored_results[:top_k]):
                debug = doc['_debug']
                print(f"   {i+1}. {doc['filename'][:60]}")
                print(f"       Final: {doc['relevance']:.3f} = Semantic({debug['semantic_score']:.3f}*0.6) + Boost({debug['keyword_boost']:.3f}) + Special({debug['student_type_bonus'] + debug['requirement_type_bonus']:.3f})")
                print(f"       Context: {debug['student_type']}/{debug['requirement_type']}, Distance: {debug['semantic_distance']:.4f}")
            
            return scored_results[:top_k]
            
        except Exception as e:
            print(f"❌ Admissions retrieval error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def retrieve_programs_documents(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        NLP-INTEGRATED specialized retrieval for programs, courses, and curriculum.
        Uses TF-IDF + Word2Vec semantic search with program-specific specialization.
        """
        import re
        
        print(f"📚 NLP-Integrated Programs Retrieval for: '{query}'")
        
        try:
            collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)
            
            # STEP 1: NLP Preprocessing
            processed_tokens = preprocess_text(query)
            query_stems = set(processed_tokens)
            print(f"📝 Preprocessed query tokens: {processed_tokens[:10]}..." if len(processed_tokens) > 10 else f"📝 Preprocessed query tokens: {processed_tokens}")
            
            # STEP 2: Generate hybrid embedding (TF-IDF + Word2Vec)
            q_emb = _embed_text(query)
            print(f"🧮 Generated hybrid embedding, dimension: {len(q_emb)}")
            
            # STEP 3: Extract specialization context
            program_info = self._extract_program_info(query)
            print(f"🎯 Extracted program info: {program_info}")
            
            # STEP 4: Get document type filters
            strategy_config = get_retrieval_strategy_config('programs_specialized')
            document_types = strategy_config.get('document_types', ['academic', 'curriculum', 'policy'])
            print(f"📋 Document type filters: {document_types}")
            
            # STEP 5: Semantic search with document type filtering
            where_clause = {
                "$and": [
                    {"source": "pdf_scrape"},
                    {"$or": [{"document_type": doc_type} for doc_type in document_types]}
                ]
            }
            
            semantic_results = collection.query(
                query_embeddings=[q_emb],
                n_results=min(top_k * 15, 60),
                include=["documents", "distances", "metadatas"],
                where=where_clause
            )
            
            ids = semantic_results.get("ids", [[]])[0]
            docs = semantic_results.get("documents", [[]])[0]
            metadatas = semantic_results.get("metadatas", [[]])[0]
            distances = semantic_results.get("distances", [[]])[0]
            
            print(f"📊 Retrieved {len(ids)} semantic candidates with document_type filtering")
            
            if not ids:
                print("⚠️ No documents found, trying without strict document_type filter")
                semantic_results = collection.query(
                    query_embeddings=[q_emb],
                    n_results=min(top_k * 15, 60),
                    include=["documents", "distances", "metadatas"],
                    where={"source": "pdf_scrape"}
                )
                ids = semantic_results.get("ids", [[]])[0]
                docs = semantic_results.get("documents", [[]])[0]
                metadatas = semantic_results.get("metadatas", [[]])[0]
                distances = semantic_results.get("distances", [[]])[0]
                print(f"📊 Retrieved {len(ids)} candidates without document_type filter")
            
            if not ids:
                return []
            
            # STEP 6: Calculate NLP-integrated scores with specialization bonuses
            scored_results = []
            
            for i, (doc_id, content, metadata, distance) in enumerate(zip(ids, docs, metadatas, distances)):
                # A. Semantic score (from TF-IDF + Word2Vec embeddings)
                semantic_score = float(1.0 / (1.0 + distance))
                
                # B. NLP keyword boost
                filename = metadata.get('filename', '')
                doc_keywords = metadata.get('keywords', '')
                boost_data = self._calculate_nlp_keyword_boost(query_stems, content, filename, doc_keywords)
                
                # C. Specialization bonuses for program matching
                program_bonus = 0.0
                year_level_bonus = 0.0
                
                # Bonus for program name match
                if program_info.get('program_name'):
                    program_name = program_info['program_name'].lower()
                    content_lower = content.lower()
                    filename_lower = filename.lower()
                    
                    if program_name in content_lower or program_name in filename_lower:
                        program_bonus = 0.08
                
                # Bonus for year level match
                if program_info.get('year_level'):
                    year_patterns = {
                        'first': ['1st', 'first', '1'],
                        'second': ['2nd', 'second', '2'],
                        'third': ['3rd', 'third', '3'],
                        'fourth': ['4th', 'fourth', '4']
                    }
                    year_level = program_info['year_level']
                    if year_level in year_patterns:
                        patterns = year_patterns[year_level]
                        content_lower = content.lower()
                        filename_lower = filename.lower()
                        if any(pattern in content_lower or pattern in filename_lower for pattern in patterns):
                            year_level_bonus = 0.05
                
                # D. FINAL SCORE: Semantic (60%) + Keyword Boost (27%) + Specialization (13%)
                final_score = (
                    semantic_score * 0.6 +
                    boost_data['total_boost'] +
                    program_bonus +
                    year_level_bonus
                )
                
                scored_results.append({
                    'id': doc_id,
                    'content': content,
                    'relevance': final_score,
                    'folder': metadata.get('folder_name', 'Unknown'),
                    'document_type': metadata.get('document_type', 'other'),
                    'target_program': metadata.get('target_program', 'all'),
                    'filename': metadata.get('filename', ''),
                    'retrieval_strategy': 'programs_nlp_integrated',
                    'current_topic': 'programs_courses',
                    '_debug': {
                        'semantic_score': semantic_score,
                        'semantic_distance': float(distance),
                        'keyword_boost': boost_data['total_boost'],
                        'content_overlap': boost_data['content_overlap'],
                        'filename_overlap': boost_data['filename_overlap'],
                        'keyword_overlap': boost_data['keyword_overlap'],
                        'program_bonus': program_bonus,
                        'year_level_bonus': year_level_bonus,
                        'program_info': program_info
                    }
                })
            
            # Sort and return top results
            scored_results.sort(key=lambda x: x['relevance'], reverse=True)
            
            print(f"✅ Top {min(top_k, len(scored_results))} NLP-integrated programs results:")
            for i, doc in enumerate(scored_results[:top_k]):
                debug = doc['_debug']
                print(f"   {i+1}. {doc['filename'][:60]}")
                print(f"       Final: {doc['relevance']:.3f} = Semantic({debug['semantic_score']:.3f}*0.6) + Boost({debug['keyword_boost']:.3f}) + Special({debug['program_bonus'] + debug['year_level_bonus']:.3f})")
                print(f"       Program: {debug['program_info'].get('program_name', 'N/A')}, Distance: {debug['semantic_distance']:.4f}")
            
            return scored_results[:top_k]
            
        except Exception as e:
            print(f"❌ Programs retrieval error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def retrieve_fees_documents(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        NLP-INTEGRATED specialized retrieval for fees, payments, and financial information.
        Uses TF-IDF + Word2Vec semantic search with fee-specific specialization.
        """
        import re
        
        print(f"💰 NLP-Integrated Fees Retrieval for: '{query}'")
        
        try:
            collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)
            
            # STEP 1: NLP Preprocessing
            processed_tokens = preprocess_text(query)
            query_stems = set(processed_tokens)
            print(f"📝 Preprocessed query tokens: {processed_tokens[:10]}..." if len(processed_tokens) > 10 else f"📝 Preprocessed query tokens: {processed_tokens}")
            
            # STEP 2: Generate hybrid embedding (TF-IDF + Word2Vec)
            q_emb = _embed_text(query)
            print(f"🧮 Generated hybrid embedding, dimension: {len(q_emb)}")
            
            # STEP 3: Extract specialization context
            fee_info = self._extract_fee_info(query)
            print(f"💳 Extracted fee info: {fee_info}")
            
            # STEP 4: Get document type filters
            strategy_config = get_retrieval_strategy_config('fees_specialized')
            document_types = strategy_config.get('document_types', ['fees', 'financial', 'policy'])
            print(f"📋 Document type filters: {document_types}")
            
            # STEP 5: Semantic search with document type filtering
            where_clause = {
                "$and": [
                    {"source": "pdf_scrape"},
                    {"$or": [{"document_type": doc_type} for doc_type in document_types]}
                ]
            }
            
            semantic_results = collection.query(
                query_embeddings=[q_emb],
                n_results=min(top_k * 15, 60),
                include=["documents", "distances", "metadatas"],
                where=where_clause
            )
            
            ids = semantic_results.get("ids", [[]])[0]
            docs = semantic_results.get("documents", [[]])[0]
            metadatas = semantic_results.get("metadatas", [[]])[0]
            distances = semantic_results.get("distances", [[]])[0]
            
            print(f"📊 Retrieved {len(ids)} semantic candidates with document_type filtering")
            
            if not ids:
                print("⚠️ No documents found, trying without strict document_type filter")
                semantic_results = collection.query(
                    query_embeddings=[q_emb],
                    n_results=min(top_k * 15, 60),
                    include=["documents", "distances", "metadatas"],
                    where={"source": "pdf_scrape"}
                )
                ids = semantic_results.get("ids", [[]])[0]
                docs = semantic_results.get("documents", [[]])[0]
                metadatas = semantic_results.get("metadatas", [[]])[0]
                distances = semantic_results.get("distances", [[]])[0]
                print(f"📊 Retrieved {len(ids)} candidates without document_type filter")
            
            if not ids:
                return []
            
            # STEP 6: Calculate NLP-integrated scores with specialization bonuses
            scored_results = []
            
            for i, (doc_id, content, metadata, distance) in enumerate(zip(ids, docs, metadatas, distances)):
                # Smart regulation filtering (skip non-financial regulations)
                doc_keywords = metadata.get('keywords', '').lower()
                filename = metadata.get('filename', '').lower()
                
                is_financial_regulation = any(term in doc_keywords or term in filename 
                                             for term in ['payment', 'fee', 'financial', 'tuition', 'billing'])
                is_regulation = 'regulation' in doc_keywords
                
                if is_regulation and not is_financial_regulation:
                    continue  # Skip non-financial regulations
                
                # A. Semantic score (from TF-IDF + Word2Vec embeddings)
                semantic_score = float(1.0 / (1.0 + distance))
                
                # B. NLP keyword boost
                boost_data = self._calculate_nlp_keyword_boost(query_stems, content, filename, doc_keywords)
                
                # C. Specialization bonuses for fee matching
                fee_type_bonus = 0.0
                program_bonus = 0.0
                payment_method_bonus = 0.0
                
                # Bonus for fee type match
                if fee_info.get('fee_type'):
                    fee_type = fee_info['fee_type'].lower()
                    content_lower = content.lower()
                    if fee_type in content_lower or fee_type in filename:
                        fee_type_bonus = 0.05
                
                # Bonus for program name match (fee for specific program)
                if fee_info.get('program_name'):
                    program_name = fee_info['program_name'].lower()
                    content_lower = content.lower()
                    if program_name in content_lower or program_name in filename:
                        program_bonus = 0.05
                
                # Bonus for payment method match
                if fee_info.get('payment_method'):
                    payment_method = fee_info['payment_method'].lower()
                    content_lower = content.lower()
                    if payment_method in content_lower:
                        payment_method_bonus = 0.03
                
                # D. FINAL SCORE: Semantic (60%) + Keyword Boost (27%) + Specialization (13%)
                final_score = (
                    semantic_score * 0.6 +
                    boost_data['total_boost'] +
                    fee_type_bonus +
                    program_bonus +
                    payment_method_bonus
                )
                
                scored_results.append({
                    'id': doc_id,
                    'content': content,
                    'relevance': final_score,
                    'folder': metadata.get('folder_name', 'Unknown'),
                    'document_type': metadata.get('document_type', 'other'),
                    'target_program': metadata.get('target_program', 'all'),
                    'filename': metadata.get('filename', ''),
                    'retrieval_strategy': 'fees_nlp_integrated',
                    'current_topic': 'fees',
                    '_debug': {
                        'semantic_score': semantic_score,
                        'semantic_distance': float(distance),
                        'keyword_boost': boost_data['total_boost'],
                        'content_overlap': boost_data['content_overlap'],
                        'filename_overlap': boost_data['filename_overlap'],
                        'keyword_overlap': boost_data['keyword_overlap'],
                        'fee_type_bonus': fee_type_bonus,
                        'program_bonus': program_bonus,
                        'payment_method_bonus': payment_method_bonus,
                        'fee_info': fee_info
                    }
                })
            
            # Sort and return top results
            scored_results.sort(key=lambda x: x['relevance'], reverse=True)
            
            print(f"✅ Top {min(top_k, len(scored_results))} NLP-integrated fees results:")
            for i, doc in enumerate(scored_results[:top_k]):
                debug = doc['_debug']
                print(f"   {i+1}. {doc['filename'][:60]}")
                print(f"       Final: {doc['relevance']:.3f} = Semantic({debug['semantic_score']:.3f}*0.6) + Boost({debug['keyword_boost']:.3f}) + Special({debug['fee_type_bonus'] + debug['program_bonus'] + debug['payment_method_bonus']:.3f})")
                print(f"       Fee type: {debug['fee_info'].get('fee_type', 'N/A')}, Distance: {debug['semantic_distance']:.4f}")
            
            return scored_results[:top_k]
            
        except Exception as e:
            print(f"❌ Fees retrieval error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def retrieve_documents_by_topic_keywords(self, query: str, topic_id: str, top_k: int = 2) -> List[Dict]:
        """
        Retrieve documents filtered by topic keywords.
        Matches document metadata keywords field against topic keywords.
        """
        import re
        
        print(f"🎯 Topic-filtered retrieval for: '{query}' (topic: {topic_id})")
        
        try:
            collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)
            
            # Get topic keywords
            topic_keywords = get_topic_keywords(topic_id)
            if not topic_keywords:
                print(f"⚠️ No keywords found for topic: {topic_id}")
                return []
            
            print(f"📝 Topic keywords: {topic_keywords}")
            
            # Get ALL documents to filter by keywords
            all_docs = collection.get(
                where={"source": "pdf_scrape"},
                include=["documents", "metadatas"]
            )
            
            all_ids = all_docs.get('ids', [])
            all_contents = all_docs.get('documents', [])
            all_metadatas = all_docs.get('metadatas', [])
            
            print(f"📚 Searching through {len(all_ids)} documents...")
            
            # Filter documents by topic keywords
            topic_filtered_candidates = []
            for i, (doc_id, content, metadata) in enumerate(zip(all_ids, all_contents, all_metadatas)):
                doc_keywords = metadata.get('keywords', '').lower()
                filename = metadata.get('filename', '').lower()
                
                # Check if document keywords match any topic keywords
                keyword_matches = 0
                matched_keywords = []
                
                for topic_keyword in topic_keywords:
                    topic_keyword_lower = topic_keyword.lower()
                    # Use word boundaries to avoid substring matches
                    pattern = r'\b' + re.escape(topic_keyword_lower) + r'\b'
                    
                    if re.search(pattern, doc_keywords) or re.search(pattern, filename):
                        keyword_matches += 1
                        matched_keywords.append(topic_keyword)
                
                # Only include documents that match at least one topic keyword
                if keyword_matches > 0:
                    topic_filtered_candidates.append({
                        'id': doc_id,
                        'content': content,
                        'metadata': metadata,
                        'topic_keyword_matches': keyword_matches,
                        'matched_keywords': matched_keywords,
                        'topic_relevance': keyword_matches / len(topic_keywords)
                    })
            
            print(f"🎯 Found {len(topic_filtered_candidates)} documents matching topic keywords")
            
            if not topic_filtered_candidates:
                print(f"❌ No documents found for topic: {topic_id}")
                return []
            
            # Now perform semantic search on ALL documents (simpler approach)
            q_emb = _embed_text(query)
            
            # Get semantic scores for all documents, then filter later
            semantic_results = collection.query(
                query_embeddings=[q_emb],
                n_results=50,  # Get more results to ensure we have matches with our filtered candidates
                include=["documents", "distances", "metadatas"],
                where={"source": "pdf_scrape"}
            )
            
            semantic_ids = semantic_results.get("ids", [[]])[0]
            semantic_distances = semantic_results.get("distances", [[]])[0]
            
            # Create semantic score lookup
            semantic_scores = {}
            for doc_id, distance in zip(semantic_ids, semantic_distances):
                semantic_scores[doc_id] = float(1.0 / (1.0 + distance))
            
            # Combine topic relevance with semantic scores
            final_results = []
            for candidate in topic_filtered_candidates:
                doc_id = candidate['id']
                topic_score = candidate['topic_relevance']
                semantic_score = semantic_scores.get(doc_id, 0.3)  # Default if not in semantic results
                
                # Weighted combination: 70% topic relevance, 30% semantic
                final_score = (topic_score * 0.7) + (semantic_score * 0.3)
                
                final_results.append({
                    'id': doc_id,
                    'content': candidate['content'],
                    'relevance': final_score,
                    'folder': candidate['metadata'].get('folder_name', 'Unknown'),
                    'document_type': candidate['metadata'].get('document_type', 'other'),
                    'target_program': candidate['metadata'].get('target_program', 'all'),
                    'filename': candidate['metadata'].get('filename', ''),
                    'retrieval_strategy': 'topic-filtered',
                    'current_topic': topic_id,
                    '_debug': {
                        'topic_score': topic_score,
                        'semantic_score': semantic_score,
                        'topic_keyword_matches': candidate['topic_keyword_matches'],
                        'matched_keywords': candidate['matched_keywords']
                    }
                })
            
            # Sort by final score
            final_results.sort(key=lambda x: x['relevance'], reverse=True)
            
            # Debug output
            print(f"✅ Top {min(top_k, len(final_results))} topic-filtered results:")
            for i, doc in enumerate(final_results[:top_k]):
                debug = doc['_debug']
                print(f"   {i+1}. {doc['filename'][:70]}")
                print(f"       Score: {doc['relevance']:.3f} (topic={debug['topic_score']:.3f}, sem={debug['semantic_score']:.3f})")
                print(f"       Matched keywords: {debug['matched_keywords']}")
            
            return final_results[:top_k]
            
        except Exception as e:
            print(f"❌ Topic-filtered retrieval error: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: Use simple topic filtering method
            print("🔄 Falling back to simple topic filtering...")
            return self.retrieve_documents_by_topic_keywords_simple(query, topic_id, top_k)

    def retrieve_documents_by_topic_specialized(self, query: str, topic_id: str, top_k: int = 2) -> List[Dict]:
        """
        Main dispatcher for topic-specialized retrieval.
        Routes to appropriate specialized function based on topic.
        """
        print(f"🎯 Dispatching specialized retrieval for topic: {topic_id}")
        
        # Topic to specialized function mapping
        TOPIC_RETRIEVERS = {
            'admissions_enrollment': self.retrieve_admissions_documents,
            'programs_courses': self.retrieve_programs_documents,
            'fees': self.retrieve_fees_documents
        }
        
        # Get the appropriate retriever function
        retriever = TOPIC_RETRIEVERS.get(topic_id)
        
        if retriever:
            print(f"✅ Using specialized retriever for {topic_id}")
            try:
                return retriever(query, top_k)
            except Exception as e:
                print(f"❌ Specialized retrieval failed for {topic_id}: {e}")
                # Fallback to simple method
                print("🔄 Falling back to simple topic filtering...")
                return self.retrieve_documents_by_topic_keywords_simple(query, topic_id, top_k)
        else:
            print(f"⚠️ No specialized retriever for {topic_id}, using simple method")
            return self.retrieve_documents_by_topic_keywords_simple(query, topic_id, top_k)

    def retrieve_documents_hybrid(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        IMPROVED NLP-INTEGRATED HYBRID RETRIEVAL:
        1. Semantic search (TF-IDF + Word2Vec embeddings) as PRIMARY ranking (70%)
        2. Keyword matching with NLP preprocessing as SECONDARY boost (30%)
        3. Consistent preprocessing applied to both query and documents
        
        This properly utilizes the TF-IDF and Word2Vec models throughout retrieval.
        """
        import re
        
        print(f"🔍 NLP-Integrated Hybrid Retrieval for: '{query}'")
        
        try:
            collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)
            
            # STEP 1: Preprocess query using NLP pipeline (stemming, stopwords, tokenization)
            processed_tokens = preprocess_text(query)
            processed_query_text = " ".join(processed_tokens)
            query_stems = set(processed_tokens)
            
            print(f"📝 Original query: '{query}'")
            if len(processed_tokens) > 10:
                print(f"📝 Preprocessed tokens: {processed_tokens[:10]}... ({len(processed_tokens)} total)")
            else:
                print(f"📝 Preprocessed tokens: {processed_tokens}")
            
            # STEP 2: Generate hybrid embedding (TF-IDF + Word2Vec)
            q_emb = _embed_text(query)
            print(f"🧮 Generated hybrid embedding (TF-IDF + Word2Vec), dimension: {len(q_emb)}")
            
            # STEP 3: Semantic search as PRIMARY retrieval (using hybrid embeddings)
            semantic_results = collection.query(
                query_embeddings=[q_emb],
                n_results=min(top_k * 10, 50),  # Get more candidates for re-ranking
                include=["documents", "distances", "metadatas"],
                where={"source": "pdf_scrape"}
            )
            
            ids = semantic_results.get("ids", [[]])[0]
            docs = semantic_results.get("documents", [[]])[0]
            metadatas = semantic_results.get("metadatas", [[]])[0]
            distances = semantic_results.get("distances", [[]])[0]
            
            print(f"📊 Retrieved {len(ids)} semantic candidates using TF-IDF + Word2Vec embeddings")
            
            if not ids:
                print("⚠️ No documents found in semantic search")
                return []
            
            # STEP 4: Calculate combined scores with NLP-based keyword boosting
            final_results = []
            
            for i, (doc_id, content, metadata, distance) in enumerate(zip(ids, docs, metadatas, distances)):
                # A. Semantic similarity score (from TF-IDF + Word2Vec hybrid embeddings)
                semantic_score = float(1.0 / (1.0 + distance))
                
                # B. Keyword boost using NLP preprocessing for fair comparison
                # Preprocess document content using the SAME NLP pipeline as query
                doc_tokens = preprocess_text(content[:5000])  # Limit content length for performance
                doc_stems = set(doc_tokens)
                
                # Calculate stem overlap (after preprocessing both query and doc)
                if query_stems:
                    stem_overlap = len(query_stems & doc_stems) / len(query_stems)
                else:
                    stem_overlap = 0.0
                
                # Preprocess and check filename matches
                filename = metadata.get('filename', '').lower()
                filename_tokens = preprocess_text(filename)
                filename_stems = set(filename_tokens)
                
                if query_stems:
                    filename_overlap = len(query_stems & filename_stems) / len(query_stems)
                else:
                    filename_overlap = 0.0
                
                # Preprocess and check metadata keywords
                doc_keywords = metadata.get('keywords', '').lower()
                keyword_tokens = preprocess_text(doc_keywords)
                keyword_stems = set(keyword_tokens)
                
                if query_stems:
                    keyword_overlap = len(query_stems & keyword_stems) / len(query_stems)
                else:
                    keyword_overlap = 0.0
                
                # C. Calculate keyword boost (max 0.3 boost to add to semantic score)
                keyword_boost = (
                    stem_overlap * 0.10 +        # Content stem overlap: up to 0.10
                    filename_overlap * 0.15 +     # Filename stem overlap: up to 0.15
                    keyword_overlap * 0.05        # Metadata keyword overlap: up to 0.05
                )
                
                # D. FINAL SCORE: Semantic (base 70%) + Keyword Boost (up to 30%)
                # Semantic score is the FOUNDATION, keywords provide a BOOST
                final_score = (semantic_score * 0.7) + keyword_boost
                
                final_results.append({
                    'id': doc_id,
                    'content': content,
                    'relevance': final_score,
                    'folder': metadata.get('folder_name', 'Unknown'),
                    'document_type': metadata.get('document_type', 'other'),
                    'target_program': metadata.get('target_program', 'all'),
                    'filename': metadata.get('filename', ''),
                    'retrieval_strategy': 'nlp-integrated-hybrid',
                    '_debug': {
                        'semantic_score': semantic_score,
                        'semantic_distance': float(distance),
                        'keyword_boost': keyword_boost,
                        'stem_overlap': stem_overlap,
                        'filename_overlap': filename_overlap,
                        'keyword_overlap': keyword_overlap,
                        'final_score': final_score,
                        'query_stems_count': len(query_stems),
                        'doc_stems_count': len(doc_stems)
                    }
                })
            
            # STEP 5: Sort by final score and return top K
            final_results.sort(key=lambda x: x['relevance'], reverse=True)
            
            # Debug output
            print(f"✅ Top {min(top_k, len(final_results))} NLP-integrated hybrid results:")
            for i, doc in enumerate(final_results[:top_k]):
                debug = doc['_debug']
                print(f"   {i+1}. {doc['filename'][:60]}")
                print(f"       Final: {debug['final_score']:.3f} = Semantic({debug['semantic_score']:.3f}*0.7) + Boost({debug['keyword_boost']:.3f})")
                print(f"       Overlaps: content={debug['stem_overlap']:.2f}, file={debug['filename_overlap']:.2f}, kw={debug['keyword_overlap']:.2f}")
                print(f"       Distance: {debug['semantic_distance']:.4f}")
            
            return final_results[:top_k]
            
        except Exception as e:
            print(f"❌ NLP-integrated retrieval error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def retrieve_generic_enhanced_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Enhanced generic retrieval that searches across ALL document types.
        Used as fallback when specialized retrieval doesn't find enough results.
        """
        print(f"🔍 Enhanced Generic retrieval for: '{query}'")
        
        try:
            collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)
            
            # Get ALL documents without document_type restriction
            all_docs = collection.get(
                where={"source": "pdf_scrape"},
                include=["documents", "metadatas"]
            )
            
            all_ids = all_docs.get('ids', [])
            all_contents = all_docs.get('documents', [])
            all_metadatas = all_docs.get('metadatas', [])
            
            print(f"📚 Searching across {len(all_ids)} documents (all types)")
            
            # Enhanced keyword matching for comprehensive coverage
            query_lower = query.lower()
            scored_results = []
            
            for i, (doc_id, content, metadata) in enumerate(zip(all_ids, all_contents, all_metadatas)):
                if not content:
                    continue
                
                content_lower = content.lower()
                filename = metadata.get('filename', '').lower()
                keywords = metadata.get('keywords', '').lower()
                
                # Multi-factor scoring
                filename_score = self._calculate_filename_relevance(filename, query_lower)
                keyword_score = self._calculate_keyword_relevance(keywords, query_lower)
                content_score = self._calculate_content_relevance(content_lower, query_lower)
                
                # Enhanced scoring for policy documents containing procedures
                if metadata.get('document_type') == 'policy':
                    if any(term in content_lower for term in ['procedure', 'process', 'how to', 'steps', 'requirements']):
                        content_score *= 1.2  # Boost policy documents with procedural content
                
                total_score = (filename_score * 0.3) + (keyword_score * 0.4) + (content_score * 0.3)
                
                if total_score > 0.1:  # Lower threshold for generic search
                    scored_results.append({
                        'id': doc_id,
                        'content': content,
                        'filename': metadata.get('filename', 'Unknown'),
                        'relevance': total_score,
                        'current_topic': 'generic_enhanced',
                        'document_type': metadata.get('document_type', 'other'),
                        '_debug': {
                            'filename_score': filename_score,
                            'keyword_score': keyword_score,
                            'content_score': content_score,
                            'doc_type': metadata.get('document_type', 'other')
                        }
                    })
            
            # Sort and return top results
            scored_results.sort(key=lambda x: x['relevance'], reverse=True)
            
            print(f"✅ Top {min(top_k, len(scored_results))} generic enhanced results:")
            for i, doc in enumerate(scored_results[:top_k]):
                debug = doc['_debug']
                print(f"   {i+1}. {doc['filename'][:60]} ({debug['doc_type']})")
                print(f"       Score: {doc['relevance']:.3f} (f={debug['filename_score']:.2f}, k={debug['keyword_score']:.2f}, c={debug['content_score']:.2f})")
            
            return scored_results[:top_k]
            
        except Exception as e:
            print(f"❌ Error in generic enhanced retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []

    def retrieve_documents(self, query: str, top_k: int = 2) -> List[Dict]:
        """Main retrieval - always use simplified hybrid"""
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
        """Enhanced retrieval with proper ChromaDB filtering and keyword boosting"""
        import re
        
        q_emb = _embed_text(query)
        try:
            collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)
            
            # Build where clause with proper $and operator for multiple conditions
            conditions = [{"source": "pdf_scrape"}]  # Base condition
            
            if folder_filter:
                conditions.append({"folder_name": folder_filter})
            if document_type_filter:
                conditions.append({"document_type": document_type_filter})
            if program_filter and program_filter != 'all':
                conditions.append({"target_program": {"$in": [program_filter, "all"]}})
            
            # Use $and only if we have multiple conditions
            if len(conditions) > 1:
                where_clause = {"$and": conditions}
            else:
                where_clause = conditions[0]
            
            print(f"🔍 ChromaDB where clause: {where_clause}")
            
            # Get MORE results for keyword re-ranking
            res = collection.query(
                query_embeddings=[q_emb],
                n_results=max(top_k * 4, 20),  # Get 4x more for re-ranking
                include=["documents", "distances", "metadatas"],
                where=where_clause
            )
            
            ids = res.get("ids", [[]])[0]
            docs = res.get("documents", [[]])[0]
            metadatas = res.get("metadatas", [[]])[0]
            distances = res.get("distances", [None])[0]
            
            # UNIVERSAL KEYWORD EXTRACTION
            query_lower = query.lower()
            stop_words = {'what', 'is', 'the', 'a', 'an', 'for', 'in', 'on', 'at', 'to', 
                          'of', 'and', 'or', 'are', 'can', 'you', 'tell', 'me', 'about',
                          'how', 'when', 'where', 'who', 'which', 'do', 'does', 'i', 'my', 'will'}
            
            query_terms = set(re.findall(r'\b[a-z0-9]{2,}\b', query_lower))
            query_terms = query_terms - stop_words
            
            print(f"🔍 Query terms: {query_terms}")
            
            out = []
            for i, (doc_id, content) in enumerate(zip(ids, docs)):
                # Semantic score
                if distances is not None and i < len(distances) and distances[i] is not None:
                    semantic_score = float(1.0 / (1.0 + distances[i]))
                else:
                    semantic_score = 1.0
                    
                metadata = metadatas[i] if i < len(metadatas) else {}
                
                # KEYWORD MATCHING
                content_lower = content.lower()
                filename_lower = metadata.get('filename', '').lower()
                keywords_lower = metadata.get('keywords', '').lower()
                
                matches = 0
                for term in query_terms:
                    if term in filename_lower:
                        matches += 1.5  # Filename = highest priority
                    elif term in keywords_lower:
                        matches += 1.3  # Keywords = medium priority
                    elif term in content_lower:
                        matches += 1.0  # Content = normal priority
                
                keyword_score = min(matches / len(query_terms), 1.0) if query_terms else 0.0
                
                # Combine: 60% semantic + 40% keyword
                final_score = (semantic_score * 0.6) + (keyword_score * 0.4)
                
                out.append({
                    "id": doc_id, 
                    "content": content, 
                    "relevance": final_score,
                    "folder": metadata.get("folder_name", "Unknown"),
                    "document_type": metadata.get("document_type", "other"),
                    "target_program": metadata.get("target_program", "all"),
                    "filename": metadata.get("filename", ""),
                    "_semantic": semantic_score,
                    "_keyword": keyword_score
                })
            
            # Sort by combined score
            out.sort(key=lambda x: x['relevance'], reverse=True)
            
            print(f"📊 Top {min(top_k, len(out))} results:")
            for i, doc in enumerate(out[:top_k]):
                print(f"   {i+1}. {doc['filename'][:60]} | score={doc['relevance']:.3f} (sem={doc['_semantic']:.3f} + kw={doc['_keyword']:.3f})")
            
            return out[:top_k]
            
        except Exception as e:
            print(f"❌ Error querying Chroma: {e}")
            import traceback
            traceback.print_exc()
            return []

    def analyze_query_intent(self, query: str) -> Dict[str, str]:
        """Enhanced query analysis to determine appropriate filters"""
        query_lower = query.lower()
        
        # Enhanced document type detection with policy inclusion
        document_type = None
        if any(word in query_lower for word in ['admission', 'apply', 'requirement', 'entrance']):
            document_type = 'admission'
        elif any(word in query_lower for word in ['enroll', 'registration', 'process']):
            document_type = 'enrollment'
        elif any(word in query_lower for word in ['scholarship', 'financial aid', 'grant']):
            document_type = 'scholarship'
        elif any(word in query_lower for word in ['program', 'course', 'degree', 'major']):
            document_type = 'academic'
        elif any(word in query_lower for word in ['fee', 'cost', 'payment', 'tuition', 'pay online', 'bank', 'gcash', 'paymaya', 'payment channel', 'payment option', 'payment method']):
            document_type = 'fees'
        elif any(word in query_lower for word in ['contact', 'phone', 'email', 'office', 'help', 'assistance']):
            document_type = 'contact'
        elif any(word in query_lower for word in ['grading', 'grade', 'qpi', 'gpa', 'appeal', 'policy', 'procedure', 'visa', 'immigration', 'international student', 'letter grade', 'grading system']):
            document_type = 'policy'  # NEW: Detect policy-related queries
        
        # Enhanced program level detection
        program_filter = None
        if any(word in query_lower for word in ['undergraduate', 'bachelor', 'college']):
            program_filter = 'undergraduate'
        elif any(word in query_lower for word in ['graduate', 'master', 'phd', 'doctoral']):
            program_filter = 'graduate'
        elif any(word in query_lower for word in ['senior high', 'shs', 'grade 11', 'grade 12']):
            program_filter = 'senior_high'
        elif any(word in query_lower for word in ['international', 'foreign', 'visa', 'immigration']):
            program_filter = 'international'  # NEW: International student detection
        
        return {
            'document_type': document_type,
            'program_filter': program_filter,
            'folder_filter': None,
            'enhanced_search': True  # Flag for enhanced search mode
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
    
    def set_session_state(self, session_id: str = None, current_topic: str = None, conversation_state: str = None, current_program: str = None):
        """Set session state for guided conversation with program tracking"""
        if session_id is not None:
            self.session_state['session_id'] = session_id
        if current_topic is not None:
            self.session_state['current_topic'] = current_topic
        if conversation_state is not None:
            self.session_state['conversation_state'] = conversation_state
        if current_program is not None:
            self.session_state['current_program'] = current_program
        
        print(f"🔄 Session state updated: topic={self.session_state.get('current_topic')}, state={self.session_state.get('conversation_state')}, program={self.session_state.get('current_program')}")
    
    def get_session_state(self):
        """Get current session state"""
        return self.session_state.copy()
    
    def reset_session(self):
        """Reset session to initial state"""
        self.session_state = {
            'current_topic': None,
            'conversation_state': CONVERSATION_STATES['TOPIC_SELECTION'],
            'session_id': self.session_state.get('session_id')  # Keep session_id
        }
        self.clear_history()
        print("🔄 Session reset to topic selection")
    
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

        # Build context from retrieved docs (use full content to preserve URLs)
        doc_context = "\n\n".join([
            f"Source: {doc.get('id','')}\n{doc['content']}"  # Use full content - no truncation
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
    
    def process_guided_conversation(self, user_input: str, action_type: str = 'message', action_data: str = None):
        """
        Process guided conversation with topic-based filtering.
        Returns response with conversation state and UI controls.
        """
        try:
            current_state = self.session_state['conversation_state']
            current_topic = self.session_state['current_topic']
            
            print(f"🎯 Processing guided conversation: state={current_state}, topic={current_topic}, action={action_type}")
            
            # Handle different action types
            if action_type == 'topic_selection':
                # User selected a topic
                topic_id = action_data
                if not get_topic_info(topic_id):
                    button_configs = get_button_configs()
                    return {
                        'error': f'Invalid topic: {topic_id}',
                        'state': current_state,
                        'buttons': button_configs['topic_selection']['buttons'],
                        'input_enabled': False,
                        'current_topic': None
                    }
                
                # Set topic and move to conversation state
                self.set_session_state(
                    current_topic=topic_id,
                    conversation_state=CONVERSATION_STATES['TOPIC_CONVERSATION']
                )
                
                topic_info = get_topic_info(topic_id)
                if not topic_info:
                    return {
                        'response': "Sorry, I couldn't find information about that topic.",
                        'state': CONVERSATION_STATES['TOPIC_SELECTION'],
                        'buttons': get_button_configs()['topic_selection']['buttons'],
                        'input_enabled': get_button_configs()['topic_selection']['input_enabled'],
                        'current_topic': None
                    }
                
                welcome_message = f"Great! You've selected **{topic_info['label']}**. {topic_info['description']}\n\nWhat would you like to know about this topic?"
                
                button_configs = get_button_configs()
                return {
                    'response': welcome_message,
                    'state': CONVERSATION_STATES['TOPIC_CONVERSATION'],
                    'buttons': button_configs['topic_conversation']['buttons'],
                    'input_enabled': button_configs['topic_conversation']['input_enabled'],
                    'current_topic': topic_id,
                    'topic_info': topic_info
                }
            
            elif action_type == 'action':
                # Handle follow-up actions
                if action_data == 'change_topic':
                    # Reset to topic selection
                    self.set_session_state(
                        current_topic=None,
                        conversation_state=CONVERSATION_STATES['TOPIC_SELECTION']
                    )
                    
                    button_configs = get_button_configs()
                    return {
                        'response': button_configs['topic_selection']['message'],
                        'state': CONVERSATION_STATES['TOPIC_SELECTION'],
                        'buttons': button_configs['topic_selection']['buttons'],
                        'input_enabled': button_configs['topic_selection']['input_enabled'],
                        'current_topic': None
                    }
            
            elif action_type == 'message':
                # Handle text message in current context
                if current_state == CONVERSATION_STATES['TOPIC_SELECTION']:
                    # User sent text when they should select a topic
                    # Try to auto-detect topic from message
                    matching_topics = find_matching_topics(user_input)
                    
                    if matching_topics and matching_topics[0]['match_count'] >= 2:
                        # Strong topic match found, auto-select it
                        best_topic = matching_topics[0]
                        topic_id = best_topic['topic_id']
                        topic_info = best_topic['topic_data']
                        
                        self.set_session_state(
                            current_topic=topic_id,
                            conversation_state=CONVERSATION_STATES['TOPIC_CONVERSATION']
                        )
                        
                        # Process the query with topic filtering
                        response, sources = self._process_topic_query(user_input, topic_id)
                        
                        button_configs = get_button_configs()
                        return {
                            'response': response,
                            'state': CONVERSATION_STATES['TOPIC_CONVERSATION'],
                            'buttons': button_configs['topic_conversation']['buttons'],
                            'input_enabled': button_configs['topic_conversation']['input_enabled'],
                            'current_topic': topic_id,
                            'sources': sources,
                            'auto_detected_topic': topic_info.get('label', 'Unknown Topic') if topic_info else 'Unknown Topic'
                        }
                    else:
                        # No clear topic match, ask user to select
                        button_configs = get_button_configs()
                        return {
                            'response': f"I understand you're asking: \"{user_input}\"\n\n{button_configs['topic_selection']['message']}",
                            'state': CONVERSATION_STATES['TOPIC_SELECTION'],
                            'buttons': button_configs['topic_selection']['buttons'],
                            'input_enabled': button_configs['topic_selection']['input_enabled'],
                            'current_topic': None
                        }
                
                elif current_state == CONVERSATION_STATES['TOPIC_CONVERSATION']:
                    # Process query within current topic
                    if not current_topic:
                        # Fallback to topic selection
                        self.set_session_state(conversation_state=CONVERSATION_STATES['TOPIC_SELECTION'])
                        button_configs = get_button_configs()
                        return {
                            'response': button_configs['topic_selection']['message'],
                            'state': CONVERSATION_STATES['TOPIC_SELECTION'],
                            'buttons': button_configs['topic_selection']['buttons'],
                            'input_enabled': button_configs['topic_selection']['input_enabled'],
                            'current_topic': None
                        }
                    
                    # Process query with topic filtering
                    response, sources = self._process_topic_query(user_input, current_topic)
                    
                    button_configs = get_button_configs()
                    return {
                        'response': response,
                        'state': CONVERSATION_STATES['TOPIC_CONVERSATION'],
                        'buttons': button_configs['topic_conversation']['buttons'],
                        'input_enabled': button_configs['topic_conversation']['input_enabled'],
                        'current_topic': current_topic,
                        'sources': sources
                    }
                
                elif current_state == CONVERSATION_STATES['FOLLOW_UP']:
                    # User sent another message after getting an answer
                    # Process as follow-up question in same topic
                    if current_topic:
                        response, sources = self._process_topic_query(user_input, current_topic)
                        
                        button_configs = get_button_configs()
                        return {
                            'response': response,
                            'state': CONVERSATION_STATES['TOPIC_CONVERSATION'],
                            'buttons': button_configs['topic_conversation']['buttons'],
                            'input_enabled': button_configs['topic_conversation']['input_enabled'],
                            'current_topic': current_topic,
                            'sources': sources
                        }
                    else:
                        # No current topic, reset to selection
                        self.set_session_state(conversation_state=CONVERSATION_STATES['TOPIC_SELECTION'])
                        button_configs = get_button_configs()
                        return {
                            'response': button_configs['topic_selection']['message'],
                            'state': CONVERSATION_STATES['TOPIC_SELECTION'],
                            'buttons': button_configs['topic_selection']['buttons'],
                            'input_enabled': button_configs['topic_selection']['input_enabled'],
                            'current_topic': None
                        }
            
            # Default fallback
            button_configs = get_button_configs()
            return {
                'response': "I'm not sure how to handle that. Let me help you select a topic.",
                'state': CONVERSATION_STATES['TOPIC_SELECTION'],
                'buttons': button_configs['topic_selection']['buttons'],
                'input_enabled': button_configs['topic_selection']['input_enabled'],
                'current_topic': None
            }
            
        except Exception as e:
            print(f"❌ Guided conversation error: {e}")
            import traceback
            traceback.print_exc()
            
            # Reset to safe state
            self.set_session_state(conversation_state=CONVERSATION_STATES['TOPIC_SELECTION'])
            button_configs = get_button_configs()
            return {
                'error': f'Processing error: {str(e)}',
                'response': button_configs['topic_selection']['message'],
                'state': CONVERSATION_STATES['TOPIC_SELECTION'],
                'buttons': button_configs['topic_selection']['buttons'],
                'input_enabled': button_configs['topic_selection']['input_enabled'],
                'current_topic': None
            }
    
    def _get_topic_specific_instructions(self, topic_id: str) -> str:
        """Get specialized instructions based on the topic"""
        if topic_id == 'admissions_enrollment':
            return """TOPIC-SPECIFIC INSTRUCTIONS FOR ADMISSIONS AND ENROLLMENT:

=== STUDENT TYPE HANDLING ===
- **DEFAULT BEHAVIOR**: If the user asks generally about "admissions" or "requirements" WITHOUT specifying a student type, provide information for **NEW STUDENTS ONLY** (first-time college students, freshmen, incoming students)
- **SPECIFIC STUDENT TYPES**: If the user mentions a specific student type, provide information ONLY for that type:
  * **NEW STUDENTS**: First-time college students, freshmen, incoming students
  * **TRANSFER STUDENTS**: Students transferring from other institutions, shifters, lateral entry
  * **INTERNATIONAL STUDENTS**: Foreign students, overseas students, non-Filipino students
  * **SCHOLAR STUDENTS**: Scholarship recipients, financial aid recipients, grant holders
- **DO NOT MIX**: Never mix information from different student types in a single response
- **FOCUS**: Cover admission requirements, required documents, processes, and procedures specific to the identified student type"""

        elif topic_id == 'programs_courses':
            return """TOPIC-SPECIFIC INSTRUCTIONS FOR PROGRAMS AND COURSES:

=== PRONOUN RESOLUTION ===
- **CRITICAL**: When user uses pronouns like "its", "it", "this", "that" referring to a program:
  * Check conversation history for the previously mentioned program
  * "What is its curriculum?" → "its" refers to the program mentioned in previous query
  * "Tell me about it" → "it" refers to the program from context
  * DO NOT interpret "its" as "IT" (Information Technology) unless explicitly stated
- **CONTEXT PRIORITY**: Always prioritize conversation context over literal interpretation of pronouns
- **CURRICULUM QUERIES**: When pronouns are used with curriculum-related terms, resolve to the program from conversation history

=== CRITICAL PROGRAM VALIDATION ===
- **ONLY ANSWER** about programs that are EXPLICITLY mentioned in the provided context documents
- **NEVER INVENT** or suggest programs that are not in the context
- **IF NO CONTEXT FOUND** for a program query, respond with: "I don't have information about that program in my knowledge base. Please check our official program list or contact admissions directly."
- **NO SPECULATION**: Do not suggest similar programs or make assumptions about program availability
- **NEVER SAY**: "However, we do offer..." or suggest alternative programs unless they are explicitly mentioned in context

=== PROGRAM AVAILABILITY QUERIES ===
- For questions like "Is [program] available?" or "Do you offer [program]?":
  * If program is in context documents: Provide information with school and cluster details
  * If program is NOT in context documents: "I don't have information about [program] in my knowledge base. For the most current list of available programs, please contact our admissions office."
- **STRICT RULE**: Only confirm programs that appear in the official program list or curriculum documents

=== PROGRAM MATCHING ===
- **BASE RESPONSES** on the specific COURSE NAME and ACRONYM mentioned by the user
- **SCOPE**: Cover UNDERGRADUATE PROGRAMS ONLY (Bachelor's degrees, BS, BA programs)
- **MATCH VARIATIONS**: If user mentions a course name (e.g., "Computer Science") or acronym (e.g., "BSCS", "BS CS", "BS COMSCI"), provide information specific to that program
- **INCLUDE**: The whole document context given for the matched program
- **AVOID**: Graduate programs, master's degrees, doctoral programs, senior high school programs

=== SCHOOL AND CLUSTER QUERIES ===
- For questions about schools, clusters, or program lists:
  * Use the official program list document as primary source
  * Organize response by School → Cluster → Programs structure
  * Include full program names and abbreviations as shown in the official list
- **STRUCTURE**: School of [Name] → [Cluster] (Cluster) → Programs with codes and full names

=== CURRICULUM QUERIES ===
- For specific program curriculum questions:
  * First confirm program exists in official list or context
  * Then provide curriculum details from curriculum documents
  * Always include links to curriculum documents when available

=== RESPONSE FORMAT ===
- For availability: Start with clear confirmation based on official documents
- For program lists: Use the exact structure from the official document
- For curriculum: Combine program confirmation + curriculum details
- Always cite sources when providing program information

=== LINK HANDLING ===
- **END WITH LINKS**: Always end your response with source links from the context documents
- **HYPERLINKS**: Format URLs as clickable hyperlinks using Markdown syntax: [link text](URL)
- **CRITICAL**: Use the EXACT URL from the document metadata - DO NOT modify, autocorrect, or change ANY part of the URL
- **NO CORRECTIONS**: Do NOT fix typos in URLs, do NOT change "Technolgy" to "Technology", do NOT modify any part of the original URL
- **PRESERVE ORIGINAL**: Copy the URL character-for-character exactly as it appears in the source document
- **FORMAT**: Use "For more information about [topic], head to this link: [link text](EXACT_URL_FROM_DOCUMENT)"
- **MANDATORY**: Links must use markdown format [text](url) to render as clickable blue hyperlinks

=== OFFICIAL PROGRAM STRUCTURE ===
**School of Arts & Sciences**:
- Humanities & Letters (Cluster): AB ENG, AB MC, AB IDS (various minors), AB PHILO
- Natural Sciences & Mathematics (Cluster): BS BIO, BS CHEM, BS MATH, BS ENVI SCI
- Computer Studies (Cluster): BS IS, BS IT, BS CS, BS DS
- Social Sciences (Cluster): AB ECON, AB POLSCI, AB PSYCH, AB SOCIO, AB IS, AB ANTHRO

**School of Business & Governance**:
- Accountancy (Cluster): BS A, BS MA
- Business Management (Cluster): BS BM, BS ENTREP, BS FIN, BS HRDM, BS MKTG, BPM

**School of Education**: BECE, BEED, BSED (English, Math, Science, Social Studies)

**School of Engineering & Architecture**: BS AE, BS ARCH, BS CHE, BS CE, BS COMP ENG, BS EE, BS ELECTRONICS ENG, BS IE, BS ME, BS RE

**School of Nursing**: BS N"""

        elif topic_id == 'fees':
            return """TOPIC-SPECIFIC INSTRUCTIONS FOR FEES:

=== FEE INFORMATION HANDLING ===
- **BASE RESPONSES** on the specific COURSE/PROGRAM mentioned by the user
- **MATCH**: Provide fee information specific to the program the user asked about
- **INCLUDE**: Tuition fees, miscellaneous fees, payment schedules, installment options for that specific program
- **DIFFERENTIATE**: Different programs may have different fee structures
- **SCOPE**: Undergraduate program fees only"""

        else:
            # Generic instructions for any other topics
            return """TOPIC-SPECIFIC INSTRUCTIONS:
- Focus on answering questions within this topic area
- Use only information from the provided context documents"""
    
    def _normalize_program_acronyms(self, query: str) -> str:
        """Normalize program acronyms to include space after BS/BA/AB (e.g., 'bsa' -> 'bs a')"""
        import re
        
        # Define common program acronyms and their variations (based on ADDU official programs)
        program_mappings = {
            # Business and Governance
            r'\bbpm\b': 'BPM',  # Public Management
            r'\bbsa\b': 'BSA',  # Accountancy
            r'\bbsma\b': 'BSMA',  # Management Accounting
            r'\bbsbm\b': 'BSBM',  # Business Management
            r'\bbsentrep\b': 'BS ENTREP',  # Entrepreneurship
            r'\bbsfin\b': 'BSFIN',  # Finance
            r'\bbshrdm\b': 'BSHRDM',  # Human Resource Development Management
            r'\bbsmktg\b': 'BS MKTG',  # Marketing
            
            # Arts and Sciences - Technology
            r'\bbsit\b': 'BS IT',  # Information Technology
            r'\bbscs\b': 'BS CS',  # Computer Science
            r'\bbsis\b': 'BS IS',  # Information Systems
            r'\bbsds\b': 'BS DS',  # Data Science
            
            # Arts and Sciences - Science
            r'\bbsbio\b': 'BS BIO',  # Biology
            r'\bbschem\b': 'BS CHEM',  # Chemistry
            r'\bbsmath\b': 'BS MATH',  # Mathematics
            r'\bbsenvisci\b': 'BS ENVISCI',  # Environmental Science
            r'\bbssocialwork\b': 'BS SOCIAL WORK',  # Social Work
            
            # Arts and Sciences - Arts
            r'\babanthro\b': 'AB ANTHRO',  # Anthropology (all tracks)
            r'\babanth\b': 'AB ANTHRO',  # Anthropology (short form)
            r'\babc\b': 'AB C',  # Communication
            r'\babcomm\b': 'AB C',  # Communication (alternate)
            r'\babds\b': 'AB DS',  # Development Studies
            r'\babecon\b': 'AB ECON',  # Economics
            r'\babel\b': 'AB EL',  # English Language
            r'\babis\b': 'AB IS',  # Interdisciplinary Studies / International Studies / Islamic Studies
            r'\babphilo\b': 'AB PHILO',  # Philosophy
            r'\babpolsci\b': 'AB POLSCI',  # Political Science
            r'\babpsych\b': 'AB PSYCH',  # Psychology
            r'\babsocio\b': 'AB SOCIO',  # Sociology
            
            # Education
            r'\bbece\b': 'BECE',  # Early Childhood Education
            r'\bbeed\b': 'BEED',  # Elementary Education
            r'\bbsed\b': 'BSED',  # Secondary Education
            
            # Engineering and Architecture
            r'\bbsae\b': 'BS AE',  # Aerospace Engineering
            r'\bbsarch\b': 'BS ARCH',  # Architecture
            r'\bbsche\b': 'BS CHE',  # Chemical Engineering
            r'\bbsce\b': 'BS CE',  # Civil Engineering
            r'\bbscompeng\b': 'BS COMP ENG',  # Computer Engineering
            r'\bbscpe\b': 'BS COMP ENG',  # Computer Engineering (alternate)
            r'\bbsee\b': 'BS EE',  # Electrical Engineering
            r'\bbselectronicseng\b': 'BS ELECTRONICS ENG',  # Electronics Engineering
            r'\bbsie\b': 'BS IE',  # Industrial Engineering
            r'\bbsme\b': 'BS ME',  # Mechanical Engineering
            r'\bbsre\b': 'BS RE',  # Robotics Engineering
            
            # Nursing
            r'\bbsn\b': 'BSN',  # Nursing
        }
        
        # Apply normalizations (case insensitive)
        normalized_query = query
        for pattern, replacement in program_mappings.items():
            normalized_query = re.sub(pattern, replacement, normalized_query, flags=re.IGNORECASE)
        
        return normalized_query

    def _is_nonsensical_query(self, query: str) -> bool:
        """Detect if the query is nonsensical, unclear, or doesn't contain meaningful content"""
        query_lower = query.lower().strip()
        
        # Check for very short queries (less than 3 characters)
        if len(query_lower) < 3:
            return True
            
        # Check for repeated characters (like "ggg", "aaa", "xxx")
        if len(set(query_lower)) <= 2 and len(query_lower) >= 3:
            return True
        
        # Check for random character sequences (no vowels pattern, too random)
        vowels = set('aeiou')
        consonants = set('bcdfghjklmnpqrstvwxyz')
        
        # If it's all consonants and longer than 6 chars, likely nonsensical
        if len(query_lower) > 6 and all(c in consonants for c in query_lower if c.isalpha()):
            return True
        
        # Check for excessive consonant-to-vowel ratio (random typing indicator)
        alpha_chars = [c for c in query_lower if c.isalpha()]
        if len(alpha_chars) > 5:
            vowel_count = sum(1 for c in alpha_chars if c in vowels)
            consonant_count = sum(1 for c in alpha_chars if c in consonants)
            
            # If less than 20% vowels in a word longer than 5 chars, likely nonsensical
            if vowel_count / len(alpha_chars) < 0.2:
                return True
            
        # Check for common nonsensical patterns
        nonsensical_patterns = [
            r'^[a-z]{1,2}$',  # Single or double letters only
            r'^[^a-zA-Z0-9\s]+$',  # Only special characters
            r'^(.)\1{2,}$',  # Repeated characters (3+ times) like "aaa", "ggg", "xxx"
        ]
        
        import re
        for pattern in nonsensical_patterns:
            if re.match(pattern, query_lower):
                return True
                
        # Check for queries that are just numbers or special characters
        if query_lower.isdigit() or not any(c.isalpha() for c in query_lower):
            return True
            
        return False

    def _is_privacy_related_query(self, query: str, topic_id: str) -> bool:
        """Detect if the query is asking for confidential/private information"""
        query_lower = query.lower().strip()
        
        # Only apply privacy checks for admissions/enrollment topic
        if topic_id != 'admissions_enrollment':
            return False
        
        # Privacy-sensitive keywords and phrases
        privacy_keywords = [
            # Grades and scores
            'my grade', 'my score', 'my result', 'my exam result',
            'what grade', 'what score', 'my stanine', 'stanine score',
            'entrance exam score', 'entrance exam result', 'exam grade',
            'test score', 'test result', 'assessment score', 'assessment result',
            
            # Passing scores/thresholds
            'passing grade', 'passing score', 'minimum score', 'minimum grade',
            'cut off', 'cutoff', 'cut-off', 'threshold', 'required score',
            'required grade', 'qualifying score', 'qualifying grade',
            
            # Personal information
            'my application', 'my status', 'application status',
            'admission status', 'acceptance status', 'my admission',
            
            # Specific score inquiries
            'what is the passing', 'what is passing', 'how much to pass',
            'score to pass', 'grade to pass', 'need to pass',
            'score needed', 'grade needed', 'minimum to pass'
        ]
        
        # Check if query contains any privacy-sensitive keywords
        for keyword in privacy_keywords:
            if keyword in query_lower:
                return True
        
        return False

    def _detect_cross_topic_query(self, query: str) -> Optional[str]:
        """Detect if user is asking about a different topic than the current one"""
        query_lower = query.lower()
        
        # Define topic detection keywords
        topic_keywords = {
            'admissions_enrollment': [
                'admission', 'requirements', 'application', 'entrance', 'apply', 'qualifying',
                'enrollment', 'registration', 'enroll', 'register', 'sign up',
                'new student', 'freshman', 'first year', 'incoming',
                'scholar', 'scholarship', 'financial aid', 'grant', 'funding',
                'transferee', 'transfer', 'shifter', 'lateral entry',
                'international', 'foreign student', 'foreign', 'overseas',
                'documents', 'documents needed', 'requirements list',
                'transcript', 'diploma', 'certificate', 'form 137', 'form 138',
                'birth certificate', 'medical certificate', 'clearance',
                'recommendation letter', 'essay', 'portfolio',
                'entrance exam', 'interview', 'assessment'
            ],
            'programs_courses': [
                'program', 'degree', 'course', 'major', 'bachelor',
                'undergraduate', 'college', 'school', 'department', 'faculty',
                'BS', 'BA', 'curriculum', 'courses', 'subjects', 'syllabus',
                
                # Business and Governance
                'bpm', 'bsa', 'bsma', 'bsbm', 'bsentrep', 'bsfin', 'bshrdm', 'bsmktg',
                'public management', 'accountancy', 'management accounting', 'business management', 
                'entrepreneurship', 'finance', 'human resource development management', 'marketing',
                
                # Arts and Sciences - Technology
                'bsit', 'bscs', 'bsis', 'bsds', 'bs it', 'bs cs', 'bs is', 'bs ds',
                'information technology', 'computer science', 'information systems', 'data science',
                
                # Arts and Sciences - Science
                'bsbio', 'bschem', 'bsmath', 'bsenvisci', 'bssocialwork', 'bs bio', 'bs chem', 'bs math', 'bs envisci', 'bs social work',
                'biology', 'chemistry', 'mathematics', 'environmental science', 'social work',
                
                # Arts and Sciences - Arts
                'abanthro', 'abanth', 'abc', 'abcomm', 'abds', 'abecon', 'abel', 'abis', 'abphilo', 'abpolsci', 'abpsych', 'absocio',
                'ab anthro', 'ab c', 'ab ds', 'ab econ', 'ab el', 'ab is', 'ab philo', 'ab polsci', 'ab psych', 'ab socio',
                'anthropology', 'communication', 'development studies', 'economics', 'english language', 
                'interdisciplinary studies', 'international studies', 'islamic studies', 'philosophy', 
                'political science', 'psychology', 'sociology',
                
                # Education
                'bece', 'beed', 'bsed', 'early childhood education', 'elementary education', 'secondary education',
                
                # Engineering and Architecture
                'bsae', 'bsarch', 'bsche', 'bsce', 'bscompeng', 'bscpe', 'bsee', 'bbselectronicseng', 'bsie', 'bsme', 'bsre',
                'bs ae', 'bs arch', 'bs che', 'bs ce', 'bs comp eng', 'bs ee', 'bs electronics eng', 'bs ie', 'bs me', 'bs re',
                'aerospace engineering', 'architecture', 'chemical engineering', 'civil engineering', 
                'computer engineering', 'electrical engineering', 'electronics engineering', 
                'industrial engineering', 'mechanical engineering', 'robotics engineering',
                
                # Nursing
                'bsn', 'nursing'
            ],
            'fees': [
                'fees', 'tuition', 'payment', 'cost', 'price', 'amount', 'billing',
                'payment plan', 'installment', 'due date', 'payment schedule',
                'down payment', 'balance', 'discount',
                'miscellaneous fees', 'laboratory fees', 'library fees',
                'graduation fee', 'examination fee', 'registration fee',
                'development fee', 'student activities fee',
                'scholarship', 'financial aid', 'grant', 'subsidy'
            ]
        }
        
        # Special handling for fee-related queries with program names
        # If the query contains fee keywords AND program keywords, prioritize fees topic
        fee_keywords = ['fees', 'tuition', 'payment', 'cost', 'price', 'amount', 'billing']
        program_keywords = [
            # Business and Governance
            'bpm', 'bsa', 'bsma', 'bsbm', 'bsentrep', 'bsfin', 'bshrdm', 'bsmktg',
            'public management', 'accountancy', 'management accounting', 'business management', 
            'entrepreneurship', 'finance', 'human resource development management', 'marketing',
            
            # Arts and Sciences - Technology
            'bsit', 'bscs', 'bsis', 'bsds', 'bs it', 'bs cs', 'bs is', 'bs ds',
            'information technology', 'computer science', 'information systems', 'data science',
            
            # Arts and Sciences - Science
            'bsbio', 'bschem', 'bsmath', 'bsenvisci', 'bssocialwork', 'bs bio', 'bs chem', 'bs math', 'bs envisci', 'bs social work',
            'biology', 'chemistry', 'mathematics', 'environmental science', 'social work',
            
            # Arts and Sciences - Arts
            'abanthro', 'abanth', 'abc', 'abcomm', 'abds', 'abecon', 'abel', 'abis', 'abphilo', 'abpolsci', 'abpsych', 'absocio',
            'ab anthro', 'ab c', 'ab ds', 'ab econ', 'ab el', 'ab is', 'ab philo', 'ab polsci', 'ab psych', 'ab socio',
            'anthropology', 'communication', 'development studies', 'economics', 'english language', 
            'interdisciplinary studies', 'international studies', 'islamic studies', 'philosophy', 
            'political science', 'psychology', 'sociology',
            
            # Education
            'bece', 'beed', 'bsed', 'early childhood education', 'elementary education', 'secondary education',
            
            # Engineering and Architecture
            'bsae', 'bsarch', 'bsche', 'bsce', 'bscompeng', 'bscpe', 'bsee', 'bbselectronicseng', 'bsie', 'bsme', 'bsre',
            'bs ae', 'bs arch', 'bs che', 'bs ce', 'bs comp eng', 'bs ee', 'bs electronics eng', 'bs ie', 'bs me', 'bs re',
            'aerospace engineering', 'architecture', 'chemical engineering', 'civil engineering', 
            'computer engineering', 'electrical engineering', 'electronics engineering', 
            'industrial engineering', 'mechanical engineering', 'robotics engineering',
            
            # Nursing
            'bsn', 'nursing'
        ]
        
        has_fee_keyword = any(keyword in query_lower for keyword in fee_keywords)
        has_program_keyword = any(keyword in query_lower for keyword in program_keywords)
        
        if has_fee_keyword and has_program_keyword:
            return 'fees'  # Prioritize fees topic for program-specific fee queries
        
        # Count keyword matches for each topic
        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            matches = sum(1 for keyword in keywords if keyword.lower() in query_lower)
            if matches > 0:
                topic_scores[topic] = matches
        
        # Return the topic with the highest score if it's significant
        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            # Only consider it a cross-topic query if there are at least 2 keyword matches
            if topic_scores[best_topic] >= 2:
                return best_topic
        
        return None
    
    def _process_topic_query(self, query: str, topic_id: str):
        """Process a query within a specific topic context with conversation history awareness"""
        try:
            # Enhanced program extraction with conversation history awareness
            program_info = self._extract_program_info_with_history(query)
            
            # CRITICAL: Preprocess pronouns if we have program context
            preprocessed_query = query
            if program_info.get('program_name') and program_info.get('context_source'):
                preprocessed_query = self._preprocess_pronoun_query(query, program_info['program_name'])
            
            # Normalize program acronyms (e.g., 'bsa' -> 'BS A')
            normalized_query = self._normalize_program_acronyms(preprocessed_query)
            
            # If we found program context from history, enhance the query
            enhanced_query = normalized_query
            if program_info.get('program_name') and program_info.get('context_source'):
                # Only enhance if the current query doesn't already contain the program name
                current_program_info = self._extract_program_info(preprocessed_query)
                if not current_program_info.get('program_name'):
                    enhanced_query = f"{program_info['program_name']} {normalized_query}"
                    print(f"🎯 Enhanced query with program context from {program_info['context_source']}: '{query}' → '{enhanced_query}'")
                else:
                    print(f"📝 Current query already contains program info, using normalized query: '{enhanced_query}'")
            else:
                print(f"📝 No program context found, using normalized query: '{enhanced_query}'")
            
            # Check if query is nonsensical or unclear (use original query for this check)
            if self._is_nonsensical_query(query):
                topic_info = get_topic_info(topic_id)
                topic_label = topic_info.get('label', topic_id) if topic_info else topic_id
                
                # Provide topic-specific examples
                if topic_id == 'admissions_enrollment':
                    examples = "- What are the admission requirements?\n- How do I apply as a new student?\n- What documents do I need?"
                elif topic_id == 'programs_courses':
                    examples = "- What programs are available?\n- Tell me about Computer Science\n- What courses are in BS IT?"
                elif topic_id == 'fees':
                    examples = "- What are the tuition fees?\n- How much does BS Computer Science cost?\n- What are the payment options?"
                else:
                    examples = "- What information do you need?\n- How can I help you?\n- What would you like to know?"
                
                response_text = f"""I'm sorry, I don't understand your query. 

Please ask a clear question about **{topic_label}**. For example:
{examples}

Try rephrasing your question with more specific details."""

                return response_text, []
            
            # Check if query is asking for private/confidential information
            if self._is_privacy_related_query(query, topic_id):
                response_text = """I apologize, but I cannot provide information about:

- **Individual exam scores or grades** (including stanine scores, entrance exam results)
- **Passing scores or cut-off grades** (these are confidential and not publicly disclosed)
- **Personal application status** (this requires accessing your personal records)
- **Specific score thresholds** (minimum/required scores are not disclosed)

**For privacy and security reasons**, this information is confidential and not disclosed publicly.


**What I can help you with:**
- General admission requirements and processes
- Required documents for application
- Application procedures and timelines
- Contact information for the Admissions Office

Feel free to ask about these general admission topics!"""

                return response_text, []
            
            # Check if user is asking about a different topic
            detected_topic = self._detect_cross_topic_query(query)
            
            if detected_topic and detected_topic != topic_id:
                # User is asking about a different topic - provide helpful guidance
                current_topic_info = get_topic_info(topic_id)
                detected_topic_info = get_topic_info(detected_topic)
                current_topic_label = current_topic_info.get('label', topic_id) if current_topic_info else topic_id
                detected_topic_label = detected_topic_info.get('label', detected_topic) if detected_topic_info else detected_topic
                
                response_text = f"""I notice you're asking about **{detected_topic_label}**, but we're currently in the **{current_topic_label}** section.

To get the most accurate information about {detected_topic_label}, please:

1. Click **"Change Topic"** below
2. Select **"{detected_topic_label}"** from the topic list
3. Ask your question again

This will ensure you get the most relevant and up-to-date information for your query."""

                return response_text, []
            
            # For programs topic, validate program availability first
            if topic_id == 'programs_courses':
                # Check if this is a program availability or list query
                availability_indicators = ['is there', 'do you have', 'available', 'offer', 'what programs', 'list of programs', 'what clusters', 'programs in', 'does addu have']
                
                if any(indicator in query.lower() for indicator in availability_indicators):
                    print(f"🔍 Detected program availability query: '{query}'")
                    
                    # First check our program mappings for quick validation
                    availability_result = self._parse_program_availability("", query)
                    
                    if availability_result['exists'] == False:
                        # Program definitely doesn't exist - provide clear response
                        response_text = "Based on our official program list, that program is not currently offered at Ateneo de Davao University. For the most up-to-date list of available programs, please contact our admissions office."
                        return response_text, []
                    
                    elif availability_result['exists'] == True:
                        # Program exists, enhance query with program context
                        program_context = f"Program: {availability_result['details']}"
                        if availability_result['school']:
                            program_context += f"\nSchool: {availability_result['school']}"
                        if availability_result['cluster']:
                            program_context += f"\nCluster: {availability_result['cluster']}"
                        
                        enhanced_query = f"{enhanced_query}\n\nProgram Context: {program_context}"
                        print(f"✅ Program exists, enhanced query with context")
                    
                    # For general program list queries, try to retrieve the program list document
                    elif any(term in query.lower() for term in ['what programs', 'list programs', 'what clusters', 'programs available']):
                        print(f"🔍 Detected program list query, retrieving official program list")
                        program_list_doc = self._retrieve_program_list_document()
                        
                        if program_list_doc['found']:
                            # Format and return the program list response
                            formatted_response = self._format_program_list_response(program_list_doc['content'], query)
                            return formatted_response, [{'id': 'program_list_official', 'content': program_list_doc['content'], 'metadata': program_list_doc['metadata']}]
                        else:
                            print("⚠️ Could not retrieve program list document")
            
            # Use specialized topic retrieval for better accuracy and efficiency (with enhanced query)
            relevant_docs = self.retrieve_documents_by_topic_specialized(enhanced_query, topic_id, top_k=3)
            
            if not relevant_docs:
                topic_info = get_topic_info(topic_id)
                topic_label = topic_info.get('label', topic_id) if topic_info else topic_id
                response_text = f"I don't have specific information about that in the {topic_label} topic. Could you try rephrasing your question?"
                
                return response_text, []
            
            # Build context from retrieved docs (use full content to preserve URLs)
            doc_context = "\n\n".join([
                f"Source: {doc.get('id','')}\n{doc['content']}"  # Use full content - no truncation
                for doc in relevant_docs[:3]
            ])
            
            # Build prompt with topic context and specialized instructions
            topic_info = get_topic_info(topic_id)
            topic_label = topic_info.get('label', topic_id) if topic_info else topic_id
            
            # Build topic-specific instructions
            topic_specific_instructions = self._get_topic_specific_instructions(topic_id)
            
            prompt = f"""<|system|>
You are an ADDU (Ateneo de Davao University) Admissions Assistant. You provide accurate, helpful information based strictly on the provided context documents.

CRITICAL URL RULE: When displaying URLs, use the EXACT URL from the source document. Do NOT modify, autocorrect, or change ANY part of URLs. If a URL contains "Technolgy" (missing 'o'), keep it as "Technolgy" - do NOT change to "Technology".

{topic_specific_instructions}

GENERAL RESPONSE RULES:
- Be direct and concise
- Use simple formatting
- No introductory phrases like "Based on the provided documentation"
- No closing phrases like "I hope this helps"
- Start directly with the answer
- Use numbered lists for steps
- Use bullet points for items
- Bold important terms only when necessary

CONTEXT MATCHING:
- Only use information that directly matches the user's specific query
- If context contains multiple student types/programs but user asked about one specific type, filter accordingly
- Prioritize exact matches over general information
</|system|>

<|context|>
{doc_context}
</|context|>

<|user|>
{enhanced_query}
</|user|>

<|assistant|>
"""
            
            # Generate response (non-streaming)
            from .together_ai_interface import stream_response_together, generate_response
            
            # Non-streaming mode - return complete response
            full_response = ""
            for chunk in stream_response_together(prompt, max_tokens=3000):
                full_response += chunk
            
            # Update session state with current program if found
            current_program_info = self._extract_program_info(query)
            if current_program_info.get('program_name'):
                # Update session state immediately if we found a program in current query
                self.set_session_state(current_program=current_program_info['program_name'])
                print(f"📝 Updated session state with current program: {current_program_info['program_name']}")
            elif program_info.get('program_name') and program_info.get('context_source') == 'query_history_1':
                # Also update session state if we found a very recent program from query history
                self.set_session_state(current_program=program_info['program_name'])
                print(f"📝 Updated session state with recent program from history: {program_info['program_name']}")
            
            # Add to history (use original query for history)
            self.add_to_history(query, full_response)
            
            return full_response, relevant_docs
            
        except Exception as e:
            print(f"❌ Topic query processing error: {e}")
            response_text = f"I encountered an error processing your question. Please try again."
            return response_text, []

    def process_query_stream(self, query: str, correct_spelling: bool = True, max_tokens: int = 5000,
                        use_history: bool = True, require_context: bool = True, 
                        min_relevance: float = 0.1) -> Generator[Dict, None, None]:
        """
        Process query and yield streaming chunks for real-time response
        Yields dictionaries with 'chunk', 'error', or 'done' keys
        """
        try:
            # Start timing
            start_time = time.time()
            
            # DISABLE typo correction - it's causing issues
            # if correct_spelling and len(query) < 50:
            #     corrected_query = correct_typos(query)
            #     if corrected_query.lower() != query.lower():
            #         query = corrected_query
            #         yield {"info": f"Corrected query: '{query}' → '{corrected_query}'"}

            # SIMPLE RETRIEVAL - No context expansion
            # expanded_query = self._expand_query_with_context(query)

            # Retrieve docs with original query
            try:
                relevant_docs = self.retrieve_documents(query)
            except Exception as e:
                yield {"error": f"Retrieval error: {e}"}
                return

            # Add debug info
            print(f"🔍 RAW RETRIEVAL DEBUG:")
            print(f"📝 Query: '{query}'")
            print(f"📄 Raw retrieved docs: {len(relevant_docs)}")
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

            # Build context from retrieved docs
            doc_context = "\n\n".join([
                f"Source: {doc.get('id','')}\n{doc['content']}"
                for doc in relevant_docs[:3]
            ])

            # History context
            history_context = ""
            if use_history and self.dialogue_history:
                base_prompt = f"Context information:\n{doc_context}\nQuestion: {query}\nInstructions: You must answer strictly and only using the context above.\nAnswer:"
                base_tokens = len(base_prompt.split())
                available_for_history = 3700 - base_tokens
                history_context = self.build_smart_history_context(query, available_for_history)
            else:
                history_context = ""

            # Build prompt
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

            # Generate streaming response
            from .together_ai_interface import stream_response_together
            
            print(f"\n🔍 DEBUG INFO:")
            print(f"📝 Query: '{query}'")
            print(f"📄 Retrieved docs: {len(relevant_docs)}")
            for i, doc in enumerate(relevant_docs):
                print(f"   Doc {i+1}: {doc['id']} (Relevance: {doc.get('relevance', 'N/A'):.3f})")
            
            full_response = ""
            for chunk in stream_response_together(prompt, max_tokens=max_tokens):
                full_response += chunk
                yield {"chunk": chunk}
            
            # Add to history
            self.add_to_history(query, full_response)
            
            total_time = time.time() - start_time
            print(f"⏱️ Total processing time: {total_time:.2f}s")
            
            yield {"done": True}
            
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            import traceback
            traceback.print_exc()
            yield {"error": f"Processing error: {e}"}


    def _retrieve_program_list_document(self) -> Dict:
        """
        Retrieve the official program list document from ChromaDB
        This document contains schools, clusters, and all available programs
        """
        try:
            from .chroma_connection import ChromaService
            collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)
            
            # Search for the program list document
            search_terms = [
                "School of Arts Sciences",
                "Humanities Letters Cluster", 
                "Computer Studies Cluster",
                "School of Business Governance",
                "School of Education",
                "School of Engineering Architecture",
                "School of Nursing",
                "AB ENG Bachelor of Arts",
                "BS IS Bachelor of Science"
            ]
            
            for search_term in search_terms:
                results = collection.query(
                    query_texts=[search_term],
                    n_results=5,
                    include=["documents", "metadatas"]
                )
                
                # Look for document that contains school/cluster structure
                for i, doc in enumerate(results.get('documents', [[]])[0]):
                    if any(indicator in doc for indicator in [
                        "School of Arts & Sciences",
                        "Humanities & Letters (Cluster)",
                        "Computer Studies (Cluster)", 
                        "AB ENG – Bachelor of Arts in English Language",
                        "BS IS – Bachelor of Science in Information Systems"
                    ]):
                        return {
                            'content': doc,
                            'metadata': results.get('metadatas', [[]])[0][i] if results.get('metadatas') else {},
                            'found': True
                        }
            
            return {'content': '', 'metadata': {}, 'found': False}
            
        except Exception as e:
            print(f"Error retrieving program list: {e}")
            return {'content': '', 'metadata': {}, 'found': False}

    def _parse_program_availability(self, program_list_content: str, query: str) -> Dict:
        """
        Parse the program list document to check program availability
        Returns: {'exists': bool, 'program_name': str, 'details': str, 'school': str, 'cluster': str}
        """
        query_lower = query.lower()
        
        # Define program mappings from the official list
        program_mappings = {
            # School of Arts & Sciences - Humanities & Letters
            'english': {'code': 'AB ENG', 'full': 'Bachelor of Arts in English Language', 'school': 'School of Arts & Sciences', 'cluster': 'Humanities & Letters'},
            'mass communication': {'code': 'AB MC', 'full': 'Bachelor of Arts in Mass Communication', 'school': 'School of Arts & Sciences', 'cluster': 'Humanities & Letters'},
            'communication': {'code': 'AB MC', 'full': 'Bachelor of Arts in Mass Communication', 'school': 'School of Arts & Sciences', 'cluster': 'Humanities & Letters'},
            'interdisciplinary studies': {'code': 'AB IDS', 'full': 'Bachelor of Arts in Interdisciplinary Studies', 'school': 'School of Arts & Sciences', 'cluster': 'Humanities & Letters'},
            'philosophy': {'code': 'AB PHILO', 'full': 'Bachelor of Arts Major in Philosophy (Pre-Law)', 'school': 'School of Arts & Sciences', 'cluster': 'Humanities & Letters'},
            
            # School of Arts & Sciences - Natural Sciences & Mathematics
            'biology': {'code': 'BS BIO', 'full': 'Bachelor of Science in Biology', 'school': 'School of Arts & Sciences', 'cluster': 'Natural Sciences & Mathematics'},
            'chemistry': {'code': 'BS CHEM', 'full': 'Bachelor of Science in Chemistry', 'school': 'School of Arts & Sciences', 'cluster': 'Natural Sciences & Mathematics'},
            'mathematics': {'code': 'BS MATH', 'full': 'Bachelor of Science in Mathematics', 'school': 'School of Arts & Sciences', 'cluster': 'Natural Sciences & Mathematics'},
            'environmental science': {'code': 'BS ENVI SCI', 'full': 'Bachelor of Science in Environmental Science', 'school': 'School of Arts & Sciences', 'cluster': 'Natural Sciences & Mathematics'},
            
            # School of Arts & Sciences - Computer Studies
            'information systems': {'code': 'BS IS', 'full': 'Bachelor of Science in Information Systems', 'school': 'School of Arts & Sciences', 'cluster': 'Computer Studies'},
            'information technology': {'code': 'BS IT', 'full': 'Bachelor of Science in Information Technology', 'school': 'School of Arts & Sciences', 'cluster': 'Computer Studies'},
            'computer science': {'code': 'BS CS', 'full': 'Bachelor of Science in Computer Science', 'school': 'School of Arts & Sciences', 'cluster': 'Computer Studies'},
            'data science': {'code': 'BS DS', 'full': 'Bachelor of Science in Data Science', 'school': 'School of Arts & Sciences', 'cluster': 'Computer Studies'},
            
            # School of Arts & Sciences - Social Sciences
            'economics': {'code': 'AB ECON', 'full': 'Bachelor of Arts Major in Economics', 'school': 'School of Arts & Sciences', 'cluster': 'Social Sciences'},
            'political studies': {'code': 'AB POLSCI', 'full': 'Bachelor of Arts Major in Political Studies', 'school': 'School of Arts & Sciences', 'cluster': 'Social Sciences'},
            'psychology': {'code': 'AB PSYCH', 'full': 'Bachelor of Arts Major in Psychology', 'school': 'School of Arts & Sciences', 'cluster': 'Social Sciences'},
            'sociology': {'code': 'AB SOCIO', 'full': 'Bachelor of Arts Major in Sociology', 'school': 'School of Arts & Sciences', 'cluster': 'Social Sciences'},
            'international studies': {'code': 'AB IS', 'full': 'Bachelor of Arts in International Studies', 'school': 'School of Arts & Sciences', 'cluster': 'Social Sciences'},
            'anthropology': {'code': 'AB ANTHRO', 'full': 'Bachelor of Arts in Anthropology', 'school': 'School of Arts & Sciences', 'cluster': 'Social Sciences'},
            
            # School of Business & Governance
            'accountancy': {'code': 'BS A', 'full': 'Bachelor of Science in Accountancy', 'school': 'School of Business & Governance', 'cluster': 'Accountancy'},
            'management accounting': {'code': 'BS MA', 'full': 'Bachelor of Science in Management Accounting', 'school': 'School of Business & Governance', 'cluster': 'Accountancy'},
            'business management': {'code': 'BS BM', 'full': 'Bachelor of Science in Business Management', 'school': 'School of Business & Governance', 'cluster': 'Business Management'},
            'entrepreneurship': {'code': 'BS ENTREP', 'full': 'Bachelor of Science in Entrepreneurship', 'school': 'School of Business & Governance', 'cluster': 'Business Management'},
            'finance': {'code': 'BS FIN', 'full': 'Bachelor of Science in Finance', 'school': 'School of Business & Governance', 'cluster': 'Business Management'},
            'human resource development': {'code': 'BS HRDM', 'full': 'Bachelor of Science in Human Resource Development and Management', 'school': 'School of Business & Governance', 'cluster': 'Business Management'},
            'marketing': {'code': 'BS MKTG', 'full': 'Bachelor of Science in Marketing', 'school': 'School of Business & Governance', 'cluster': 'Business Management'},
            'public management': {'code': 'BPM', 'full': 'Bachelor of Public Management', 'school': 'School of Business & Governance', 'cluster': 'Business Management'},
            
            # School of Education
            'early childhood education': {'code': 'BECE', 'full': 'Bachelor of Early Childhood Education', 'school': 'School of Education', 'cluster': 'Education'},
            'elementary education': {'code': 'BEED', 'full': 'Bachelor of Elementary Education', 'school': 'School of Education', 'cluster': 'Education'},
            'secondary education': {'code': 'BSED', 'full': 'Bachelor of Secondary Education', 'school': 'School of Education', 'cluster': 'Education'},
            
            # School of Engineering & Architecture
            'aerospace engineering': {'code': 'BS AE', 'full': 'Bachelor of Science in Aerospace Engineering', 'school': 'School of Engineering & Architecture', 'cluster': 'Engineering & Architecture'},
            'architecture': {'code': 'BS ARCH', 'full': 'Bachelor of Science in Architecture', 'school': 'School of Engineering & Architecture', 'cluster': 'Engineering & Architecture'},
            'chemical engineering': {'code': 'BS CHE', 'full': 'Bachelor of Science in Chemical Engineering', 'school': 'School of Engineering & Architecture', 'cluster': 'Engineering & Architecture'},
            'civil engineering': {'code': 'BS CE', 'full': 'Bachelor of Science in Civil Engineering', 'school': 'School of Engineering & Architecture', 'cluster': 'Engineering & Architecture'},
            'computer engineering': {'code': 'BS COMP ENG', 'full': 'Bachelor of Science in Computer Engineering', 'school': 'School of Engineering & Architecture', 'cluster': 'Engineering & Architecture'},
            'electrical engineering': {'code': 'BS EE', 'full': 'Bachelor of Science in Electrical Engineering', 'school': 'School of Engineering & Architecture', 'cluster': 'Engineering & Architecture'},
            'electronics engineering': {'code': 'BS ELECTRONICS ENG', 'full': 'Bachelor of Science in Electronics Engineering', 'school': 'School of Engineering & Architecture', 'cluster': 'Engineering & Architecture'},
            'industrial engineering': {'code': 'BS IE', 'full': 'Bachelor of Science in Industrial Engineering', 'school': 'School of Engineering & Architecture', 'cluster': 'Engineering & Architecture'},
            'mechanical engineering': {'code': 'BS ME', 'full': 'Bachelor of Science in Mechanical Engineering', 'school': 'School of Engineering & Architecture', 'cluster': 'Engineering & Architecture'},
            'robotics engineering': {'code': 'BS RE', 'full': 'Bachelor of Science in Robotics Engineering', 'school': 'School of Engineering & Architecture', 'cluster': 'Engineering & Architecture'},
            
            # School of Nursing
            'nursing': {'code': 'BS N', 'full': 'Bachelor of Science in Nursing', 'school': 'School of Nursing', 'cluster': 'Nursing'}
        }
        
        # Check for program matches
        for program_key, program_info in program_mappings.items():
            # Check various forms of the program name
            program_variations = [
                program_key,
                program_info['code'].lower(),
                program_info['code'].lower().replace(' ', ''),
                program_key.replace(' ', ''),
                program_key.replace(' ', '-')
            ]
            
            if any(variation in query_lower for variation in program_variations):
                return {
                    'exists': True,
                    'program_name': program_info['full'],
                    'code': program_info['code'],
                    'details': f"{program_info['code']} – {program_info['full']}",
                    'school': program_info['school'],
                    'cluster': program_info['cluster']
                }
        
        # If no match found, program doesn't exist
        return {'exists': False, 'program_name': None, 'details': None, 'school': None, 'cluster': None}

    def _format_program_list_response(self, program_list_content: str, query: str) -> str:
        """
        Format the program list document content based on the query type
        """
        query_lower = query.lower()
        
        # Detect query type
        if 'cluster' in query_lower:
            return self._extract_cluster_info(program_list_content, query_lower)
        elif any(school in query_lower for school in ['arts', 'sciences', 'business', 'engineering', 'education', 'nursing']):
            return self._extract_school_info(program_list_content, query_lower)
        else:
            # General program list
            return f"Here are the available undergraduate programs at Ateneo de Davao University:\n\n{program_list_content}"

    def _extract_cluster_info(self, content: str, query: str) -> str:
        """Extract cluster-specific information"""
        lines = content.split('\n')
        result = []
        in_target_cluster = False
        current_school = None
        
        cluster_keywords = {
            'computer': 'Computer Studies',
            'humanities': 'Humanities & Letters', 
            'natural': 'Natural Sciences & Mathematics',
            'sciences': 'Natural Sciences & Mathematics',
            'social': 'Social Sciences',
            'business': 'Business Management',
            'accountancy': 'Accountancy',
            'education': 'Education',
            'engineering': 'Engineering & Architecture',
            'architecture': 'Engineering & Architecture',
            'nursing': 'Nursing'
        }
        
        target_cluster = None
        for keyword, cluster_name in cluster_keywords.items():
            if keyword in query:
                target_cluster = cluster_name
                break
        
        for line in lines:
            line = line.strip()
            if line.startswith('School of'):
                current_school = line
            elif '(Cluster)' in line:
                if target_cluster and target_cluster in line:
                    in_target_cluster = True
                    result.append(f"\n{current_school}")
                    result.append(f"● {line}")
                else:
                    in_target_cluster = False
            elif in_target_cluster and line and (line[0].isdigit() or line.startswith('●')):
                result.append(line)
        
        return '\n'.join(result) if result else "I couldn't find specific cluster information for your query."

    def _extract_school_info(self, content: str, query: str) -> str:
        """Extract school-specific information"""
        lines = content.split('\n')
        result = []
        in_target_school = False
        
        school_keywords = {
            'arts': 'School of Arts & Sciences',
            'sciences': 'School of Arts & Sciences',
            'business': 'School of Business & Governance',
            'governance': 'School of Business & Governance',
            'education': 'School of Education',
            'engineering': 'School of Engineering & Architecture',
            'architecture': 'School of Engineering & Architecture',
            'nursing': 'School of Nursing'
        }
        
        target_school = None
        for keyword, school_name in school_keywords.items():
            if keyword in query:
                target_school = school_name
                break
        
        for line in lines:
            line = line.strip()
            if line.startswith('School of'):
                if target_school and target_school in line:
                    in_target_school = True
                    result.append(line)
                else:
                    in_target_school = False
            elif in_target_school and line:
                result.append(line)
        
        return '\n'.join(result) if result else f"I couldn't find information about that school."

    def _preprocess_pronoun_query(self, query: str, program_context: str = None) -> str:
        """Preprocess queries with pronouns to avoid misinterpretation"""
        if not program_context:
            return query
            
        query_lower = query.lower().strip()
        
        # If query contains pronouns and we have program context, substitute them
        pronoun_substitutions = {
            r'\bits\b': program_context,
            r'\bit\b(?!\s+is|\s+was|\s+has|\s+will|\s+can|\s+should)': program_context,  # Avoid "it is", "it was", etc.
            r'\bthis\b': program_context,
            r'\bthat\b': program_context,
            r'\bthe program\b': program_context,
            r'\bthe course\b': program_context
        }
        
        import re
        processed_query = query
        substitution_made = False
        
        for pattern, replacement in pronoun_substitutions.items():
            new_query = re.sub(pattern, replacement, processed_query, flags=re.IGNORECASE)
            if new_query != processed_query:
                processed_query = new_query
                substitution_made = True
        
        if substitution_made:
            print(f"🔄 Preprocessed pronoun query: '{query}' → '{processed_query}'")
            return processed_query
        
        return query

def test_fast_hybrid_chatbot_together():
    """Test the fast hybrid chatbot with Together AI"""
    chatbot = FastHybridChatbotTogether(use_chroma=True)
    
    print("\n⚡🔍 FAST HYBRID CHATBOT WITH TOGETHER AI ⚡🔍")
    print("Type 'exit' to quit\n")

    max_tokens = 500  # Higher default for complete responses
