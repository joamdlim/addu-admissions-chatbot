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
    print("[OK] Gensim Word2Vec available")
except ImportError:
    WORD2VEC_AVAILABLE = False
    print("[WARN] Gensim Word2Vec not available, using placeholder")

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
                 word2vec_path=None, use_chroma: bool = False, chroma_collection_name: Optional[str] = None,
                 use_hybrid_topic_retrieval: bool = True):
        # Get the current script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set default paths relative to the current script
        if embeddings_dir is None:
            embeddings_dir = os.path.join(current_dir, "embeddings")
        if processed_dir is None:
            processed_dir = os.path.join(current_dir, "processed")
        if word2vec_path is None:
            # Word2Vec model is in backend/model/, not backend/chatbot/model/
            backend_dir = os.path.dirname(current_dir)  # Go up from chatbot/ to backend/
            word2vec_path = os.path.join(backend_dir, "model", "GoogleNews-vectors-negative300.bin")
        
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
        
        # Initialize normalization config cache
        self._normalization_config = None
        self._config_file_path = os.path.join(current_dir, "program_normalization_config.json")
        self.max_history_length = 5  # Keep last 5 exchanges
        
        # Session state for guided conversation
        self.session_state = {
            'current_topic': None,
            'conversation_state': CONVERSATION_STATES['TOPIC_SELECTION'],
            'session_id': None
        }
        
        self.use_chroma = use_chroma
        self.chroma_collection_name = chroma_collection_name or os.getenv("CHROMA_COLLECTION", "documents")
        self.use_hybrid_topic_retrieval = use_hybrid_topic_retrieval

        # Dynamic program detection cache
        self._program_cache = {}
        self._filename_patterns_cache = {}
        self._last_cache_update = 0

        if self.use_chroma:
            _init_embed_models()  # ensure the TF‑IDF + Word2Vec embedder is ready
            self._init_tfidf_for_chroma()  # Initialize TF-IDF vectorizer for ChromaDB
            self._discover_programs_from_data()  # Initialize dynamic program detection
        else:
            self._load_data()
        
        # Start timing
        start_time = time.time()
        
        # Try to load Word2Vec model if available
        if WORD2VEC_AVAILABLE and os.path.exists(word2vec_path):
            try:
                print(f"[LOADING] Loading Word2Vec model from {word2vec_path}")
                self.word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
                print("[OK] Word2Vec model loaded successfully")
            except Exception as e:
                print(f"[WARNING] Failed to load Word2Vec model: {e}")
        
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
                print(f"[OK] Loaded metadata for {len(self.documents)} documents")
            else:
                print(f"[WARNING] Metadata file not found at: {self.metadata_path}")
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
                        print(f"[OK] Loaded metadata for {len(self.documents)} documents")
                        self.metadata_path = path  # Update the path
                        break
                else:
                    print("[ERROR] Could not find metadata.json in any location")
                    return
            
            # Load vectors if they exist
            if os.path.exists(self.vectors_path):
                self.vectors = np.load(self.vectors_path)
                print(f"[OK] Loaded vectors with shape: {self.vectors.shape}")
                
                # Build TF-IDF vectorizer on document content
                corpus = [" ".join(preprocess_text(doc["content"])) for doc in self.documents]
                self.tfidf_vectorizer = TfidfVectorizer()
                self.tfidf_vectorizer.fit(corpus)
                print("[OK] Built TF-IDF vectorizer")
            else:
                print("[WARNING] Vector file not found, falling back to keyword search")
        except Exception as e:
            print(f"[ERROR] Error loading data: {e}")
    
    def _init_tfidf_for_chroma(self):
        """Initialize TF-IDF vectorizer when using ChromaDB"""
        try:
            from .chroma_connection import ChromaService
            
            print("[INIT] Initializing TF-IDF vectorizer for ChromaDB...")
            
            # Get all documents from ChromaDB to build TF-IDF corpus
            collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)
            
            # Get all documents
            all_docs = collection.get(
                where={"source": "pdf_scrape"},
                include=["documents"]
            )
            
            all_contents = all_docs.get('documents', [])
            
            if all_contents:
                print(f"📚 Building TF-IDF vectorizer from {len(all_contents)} documents...")
                
                # Preprocess all documents for TF-IDF
                corpus = []
                for content in all_contents:
                    processed_content = preprocess_text(content)
                    corpus.append(" ".join(processed_content))
                
                # Build and fit TF-IDF vectorizer
                self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
                self.tfidf_vectorizer.fit(corpus)
                
                print(f"[OK] TF-IDF vectorizer initialized with {len(self.tfidf_vectorizer.vocabulary_)} features")
            else:
                print("[WARNING] No documents found in ChromaDB for TF-IDF initialization")
                
        except Exception as e:
            print(f"[ERROR] Error initializing TF-IDF for ChromaDB: {e}")
            self.tfidf_vectorizer = None
    
    def _vectorize_query(self, query: str) -> np.ndarray:
        """Convert query to vector representation with proper preprocessing for each model"""
        
        # Dual preprocessing approach for optimal performance
        # TF-IDF: Use stemmed text (consistent with training corpus)
        processed_query_stemmed = preprocess_text(query)
        query_text_stemmed = " ".join(processed_query_stemmed)
        
        # Word2Vec: Use raw text (no stemming, as Word2Vec expects full words)
        query_text_raw = query.lower()
        raw_tokens = query_text_raw.split()
        
        # Get TF-IDF vector (using stemmed text)
        try:
            if self.tfidf_vectorizer is None:
                print("⚠️ TF-IDF vectorizer not available")
                # Estimate a reasonable dimension for TF-IDF
                tfidf_dim = 1000 if self.vectors is None else self.vectors.shape[1] - 300
                tfidf_vector = np.zeros(tfidf_dim)
            else:
                tfidf_vector = sparse_to_array(self.tfidf_vectorizer.transform([query_text_stemmed]))[0]
        except Exception as e:
            print(f"❌ Error creating TF-IDF vector: {e}")
            # Create a zero vector with a reasonable dimension
            tfidf_dim = 1000 if self.vectors is None else self.vectors.shape[1] - 300
            tfidf_vector = np.zeros(tfidf_dim)
        
        # Get Word2Vec vector (using raw tokens, no stemming)
        w2v_vector = compute_word2vec_vector(raw_tokens, self.word2vec_model)
        
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
            'business administration': ['business administration', 'bsba', 'bs ba', 'business admin'],
            'entrepreneurship': ['entrepreneurship', 'bs entrep', 'bsentrep', 'entrepreneur'],
            'finance': ['finance', 'bsfin', 'bs fin'],
            'human resource development': ['human resource', 'hrdm', 'bshrdm', 'bs hrdm', 'hr'],
            'marketing': ['marketing', 'bs mktg', 'bsmktg'],
            'public management': ['public management', 'bpm', 'governance'],
            
            # Technology Programs
            'computer science': ['computer science', 'cs', 'bscs', 'bs cs', 'compsci', 'comsci'],
            'information technology': ['information technology', 'bsit', 'bs it', 'infotech'],
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
            'nursing': ['nursing', 'bsn', 'bs n', 'nurse']
        }

    def _extract_program_info(self, query: str) -> dict:
        """Extract program information from query using configurable normalization system"""
        import re
        # Remove punctuation for better program matching
        query_clean = re.sub(r'[^\w\s]', '', query)
        query_lower = query_clean.lower()
        
        program_info = {
            'program_name': None,
            'degree_level': None,
            'year_level': None,
            'course_code': None,
            'context_source': None
        }
        
        # Use the configurable normalization system for program detection
        abbreviations = self._get_program_abbreviations()
        
        import re
        matches = []
        
        # Find all potential program matches using the same logic as normalization
        for abbrev, abbrev_data in abbreviations.items():
            if not isinstance(abbrev_data, dict):
                continue
                
            full_name = abbrev_data.get("full_name", "")
            description = abbrev_data.get("description", "")
            priority = abbrev_data.get("priority", "safe")
            is_common_word = abbrev_data.get("is_common_word", False)
            
            # Calculate match score using the same logic as program availability
            match_score = self._calculate_program_match_score(query_lower, abbrev, full_name, description)
            
            if match_score > 0:
                # Apply priority filtering (same as normalization)
                should_include = False
                
                if priority == "safe":
                    should_include = True
                elif priority == "context_aware":
                    # Check for program context
                    context_patterns = self._get_context_patterns()
                    program_keywords = context_patterns.get("program_keywords", [])
                    has_program_context = any(re.search(rf'\b{keyword}\b', query_lower) for keyword in program_keywords)
                    should_include = has_program_context
                elif priority == "context_required":
                    # Only include with strong context (for common English words)
                    context_patterns = self._get_context_patterns()
                    strong_program_patterns = context_patterns.get("strong_program", [])
                    has_strong_context = False
                    
                    for pattern_template in strong_program_patterns:
                        pattern = pattern_template.replace("{abbrev}", re.escape(abbrev))
                        if re.search(pattern, query_lower):
                            has_strong_context = True
                            break
                    
                    # For common English words, be extra conservative
                    if is_common_word:
                        problematic_patterns = context_patterns.get("problematic", [])
                        has_problematic_context = any(re.search(pattern, query_lower) for pattern in problematic_patterns)
                        
                        # SMART DETECTION: Check if the abbreviation appears as uppercase in original query
                        # This handles cases like "is there IS" where "IS" is clearly a program reference
                        has_uppercase_abbrev = abbrev.upper() in query
                        
                        # If the abbreviation appears in uppercase, it's likely a program reference
                        if has_uppercase_abbrev:
                            should_include = True  # Override filtering for uppercase program references
                        else:
                            should_include = has_strong_context and not has_problematic_context
                    else:
                        should_include = has_strong_context
                
                if should_include:
                    matches.append({
                        'score': match_score,
                        'abbrev': abbrev,
                        'full_name': full_name,
                        'description': description,
                        'abbrev_data': abbrev_data
                    })
        
        # Sort matches and handle ambiguity
        if matches:
            # Sort by score with tie-breaking (same as program availability)
            def tie_breaker(match):
                score = match['score']
                length = len(match['abbrev'])
                abbrev = match['abbrev']
                
                # Apply configurable tie-breaking logic
                # Check for conflicts in the config
                abbrev_data = match.get('abbrev_data', {})
                conflicts_with = abbrev_data.get('conflicts_with', [])
                
                if conflicts_with and score >= 0.9 and length == 2:
                    # Check if any conflicting abbreviations are present
                    conflicting_abbrevs = [m['abbrev'] for m in matches if m['abbrev'] in conflicts_with]
                    
                    if conflicting_abbrevs:
                        # Give priority to more specific matches (longer abbreviations first)
                        # If same length, use alphabetical order for consistency
                        return (score, length, 1 if abbrev > min(conflicting_abbrevs) else 0)
                
                return (score, length, 0)
            
            sorted_matches = sorted(matches, key=tie_breaker, reverse=True)
            best_match = sorted_matches[0]
            
            # For fees and programs, use the normalized full_name (e.g., "BS A") instead of field name
            # This ensures consistency with the JSON config normalization
            program_info['program_name'] = best_match['full_name'] or best_match['abbrev']
        
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
        import re
        
        # Use word boundaries for precise pronoun detection
        pronoun_patterns = [
            r'\bits\b',      # "its" as standalone word
            r'\bit\b(?!\s+is|\s+was|\s+has|\s+will|\s+can|\s+should)',  # "it" but not "it is", "it was", etc.
            r'\bthis\b(?!\s+is|\s+program|\s+course)',  # "this" but not "this is", "this program"
            r'\bthat\b(?!\s+is|\s+program|\s+course)',  # "that" but not "that is", "that program"
            r'\bthe program\b',
            r'\bthe course\b'
        ]
        
        curriculum_indicators = ['curriculum', 'courses', 'subjects', 'syllabus', 'course outline', 'academic plan']
        general_program_indicators = ['tell me about', 'about it', 'information about', 'details about']
        
        # Check for pronoun context first - use regex for precise matching
        has_pronoun = any(re.search(pattern, query_lower) for pattern in pronoun_patterns)
        has_curriculum_context = any(term in query_lower for term in curriculum_indicators)
        has_general_program_context = any(term in query_lower for term in general_program_indicators)
        
        # Debug logging
        if has_pronoun:
            matched_pronouns = [pattern for pattern in pronoun_patterns if re.search(pattern, query_lower)]
            print(f"🔍 DEBUG: Detected pronouns in '{query}': {matched_pronouns}")
        if has_curriculum_context:
            matched_curriculum = [term for term in curriculum_indicators if term in query_lower]
            print(f"📚 DEBUG: Detected curriculum context: {matched_curriculum}")
        if has_general_program_context:
            matched_general = [term for term in general_program_indicators if term in query_lower]
            print(f"🎯 DEBUG: Detected general program context: {matched_general}")
        
        # Check if apparent "pronouns" are actually uppercase program abbreviations
        uppercase_program_detected = False
        if has_pronoun:
            # Check for uppercase program abbreviations that might look like pronouns
            import re
            potential_programs = re.findall(r'\b[A-Z]{2,4}\b', query)  # Find 2-4 letter uppercase words
            if potential_programs:
                # Check if any of these are actual program abbreviations
                abbreviations = self._get_program_abbreviations()
                for abbrev_upper in potential_programs:
                    abbrev_lower = abbrev_upper.lower()
                    if abbrev_lower in abbreviations:
                        abbrev_data = abbreviations[abbrev_lower]
                        if abbrev_data.get('is_common_word', False):
                            print(f"🎯 Detected uppercase program abbreviation '{abbrev_upper}' (not a pronoun)")
                            uppercase_program_detected = True
                            break
        
        # If this looks like a pronoun reference AND not an uppercase program, skip normal program extraction and go to history
        if has_pronoun and (has_curriculum_context or has_general_program_context) and not uppercase_program_detected:
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
            
            # If pronoun resolution failed, mark it as ambiguous
            print(f"❌ Pronoun resolution failed - no conversation history or session context")
            program_info['context_source'] = 'pronoun_resolution_failed'
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
            'program_level': None,
            'program_name': None
        }
        
        # Detect fee types
        fee_types = {
            'tuition': ['tuition', 'tuition fee'],
            'miscellaneous': ['miscellaneous', 'misc fee', 'other fees'],
            'laboratory': ['laboratory', 'lab fee'],
            'registration': ['registration', 'enrollment fee'],
            'graduation': ['graduation', 'graduation fee']
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
        
        # Detect program level for fee differentiation
        if any(term in query_lower for term in ['undergraduate', 'bachelor', 'bs ', 'bsa', 'bsb', 'bsc', 'bsd', 'bse', 'bsf', 'bsg', 'bsh', 'bsi', 'bsj', 'bsk', 'bsl', 'bsm', 'bsn', 'bso', 'bsp', 'bsq', 'bsr', 'bss', 'bst', 'bsu', 'bsv', 'bsw', 'bsx', 'bsy', 'bsz']):
            fee_info['program_level'] = 'undergraduate'
        elif any(term in query_lower for term in ['graduate', 'master', 'ms ', 'ma ', 'phd', 'doctorate']):
            fee_info['program_level'] = 'graduate'
        
        # ENHANCED: Use the same sophisticated JSON-based program extraction as programs
        # This provides better accuracy with normalization, context awareness, and priority handling
        program_info = self._extract_program_info(query)
        if program_info.get('program_name'):
            fee_info['program_name'] = program_info['program_name']
            print(f"📚 Enhanced program extraction for fees: '{program_info['program_name']}'")
        else:
            # Fallback to the old method if JSON-based extraction fails
            program_keywords = self._get_comprehensive_program_keywords()
            sorted_keywords = sorted(program_keywords, key=len, reverse=True)
            
            for program in sorted_keywords:
                import re
                pattern = r'\b' + re.escape(program.lower()) + r'\b'
                if re.search(pattern, query_lower):
                    fee_info['program_name'] = program
                    print(f"📚 Fallback program extraction for fees: '{program}'")
                    break
        
        return fee_info
    
    def _get_comprehensive_program_keywords(self) -> List[str]:
        """Get comprehensive list of program keywords including abbreviations"""
        return [
            # Business and Governance
            'bpm', 'bsa', 'bsma', 'bsbm', 'bsentrep', 'bsfin', 'bshrdm', 'bsmktg',
            'public management', 'accountancy', 'management accounting', 'business management', 
            'entrepreneurship', 'finance', 'human resource development management', 'marketing',
            
            # Arts and Sciences - Technology (with abbreviations)
            'bsit', 'bscs', 'bsis', 'bsds', 'bs it', 'bs cs', 'bs is', 'bs ds',
            'information technology', 'computer science', 'information systems', 'data science',
            # Abbreviations for Technology programs
            'it', 'cs', 'is', 'ds', 'compsci', 'infotech', 'datasci',
            
            # Arts and Sciences - Science (with abbreviations)
            'bsbio', 'bschem', 'bsmath', 'bsenvisci', 'bssocialwork', 'bs bio', 'bs chem', 'bs math', 'bs envisci', 'bs social work',
            'biology', 'chemistry', 'mathematics', 'environmental science', 'social work',
            # Abbreviations for Science programs
            'bio', 'chem', 'math', 'envisci', 'socialwork',
            
            # Arts and Sciences - Arts (with abbreviations)
            'abanthro', 'abanth', 'abc', 'abcomm', 'abds', 'abecon', 'abel', 'abis', 'abphilo', 'abpolsci', 'abpsych', 'absocio',
            'ab anthro', 'ab c', 'ab ds', 'ab econ', 'ab el', 'ab is', 'ab philo', 'ab polsci', 'ab psych', 'ab socio',
            'anthropology', 'communication', 'development studies', 'economics', 'english language', 
            'interdisciplinary studies', 'international studies', 'islamic studies', 'philosophy', 
            'political science', 'psychology', 'sociology',
            # Abbreviations for Arts programs
            'anthro', 'anth', 'comm', 'econ', 'el', 'philo', 'polsci', 'psych', 'socio',
            
            # Education (with abbreviations)
            'bece', 'beed', 'bsed', 'early childhood education', 'elementary education', 'secondary education',
            # Abbreviations for Education programs
            'ece', 'eed', 'sed',
            
            # Engineering and Architecture (with abbreviations)
            'bsae', 'bsarch', 'bsche', 'bsce', 'bscompeng', 'bscpe', 'bsee', 'bbselectronicseng', 'bsie', 'bsme', 'bsre',
            'bs ae', 'bs arch', 'bs che', 'bs ce', 'bs comp eng', 'bs ee', 'bs electronics eng', 'bs ie', 'bs me', 'bs re',
            'aerospace engineering', 'architecture', 'chemical engineering', 'civil engineering', 
            'computer engineering', 'electrical engineering', 'electronics engineering', 
            'industrial engineering', 'mechanical engineering', 'robotics engineering',
            # Abbreviations for Engineering programs
            'ae', 'arch', 'che', 'ce', 'compeng', 'cpe', 'ee', 'electronicseng', 'ie', 'me', 're',
            'aerospace', 'chemical', 'civil', 'computer', 'electrical', 'electronics', 'industrial', 'mechanical', 'robotics',
            
            # Nursing (with abbreviations)
            'bsn', 'nursing', 'nurse'
        ]

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
    
    def _calculate_programs_filename_score(self, filename: str, program_info: dict, query_intent: str = None) -> float:
        """Calculate filename score for programs documents"""
        score = 0.0
        
        # NEGATIVE FILTERING: Exclude irrelevant documents for curriculum queries AND program-specific subject mapping
        # Skip negative filtering only for general subject mapping queries (no specific program)
        should_apply_negative_filtering = (
            program_info.get('program_name') and 
            (query_intent != 'subject_mapping' or program_info.get('program_name'))
        )
        if should_apply_negative_filtering:
            # If looking for a specific program, exclude documents that are clearly for other programs
            irrelevant_patterns = [
                'secondary education',  # Exclude secondary education when looking for BS programs
                'bachelor of secondary education',
                'master of',  # Exclude master's programs when looking for bachelor's
                'doctor of',  # Exclude doctoral programs when looking for bachelor's
            ]
            
            program_name = program_info['program_name'].lower()
            # Only apply negative filtering if we're looking for a bachelor's program
            if program_name.startswith('bs ') or program_name.startswith('ab '):
                for pattern in irrelevant_patterns:
                    if pattern in filename.lower():
                        print(f"🚫 Excluding irrelevant document: {filename} (matches pattern: {pattern})")
                        return 0.0  # Completely exclude this document
        
        # Base score for program-related filenames
        if any(term in filename for term in ['curriculum', 'program', 'course', 'syllabus']):
            score += 0.4
        
        # CONDITIONAL PROGRAM NAME BONUS: Reduce for subject mapping to be more flexible like fees
        if program_info['program_name']:
            program_name = program_info['program_name']
            program_name_lower = program_name.lower()
            filename_lower = filename.lower()
            
            # Adjust bonus based on query intent and specificity
            if query_intent == 'subject_mapping':
                # Check if it's a program-specific subject query vs general subject query
                if program_info.get('program_name'):
                    # Program-specific subject mapping - use STRICT matching (like curriculum)
                    direct_bonus = 0.6  # Restore full bonus for program-specific queries
                    abbreviated_bonus = 0.6
                    expanded_bonus = 0.5
                    print(f"📊 Using strict program bonuses for program-specific subject mapping")
                else:
                    # General subject mapping - use FLEXIBLE matching (like fees)
                    direct_bonus = 0.3  # Keep reduced bonus for general queries
                    abbreviated_bonus = 0.3
                    expanded_bonus = 0.25
                    print(f"📊 Using flexible program bonuses for general subject mapping")
            else:
                # Full bonuses for curriculum and other queries
                direct_bonus = 0.6
                abbreviated_bonus = 0.6
                expanded_bonus = 0.5
            
            # Direct match (e.g., "BS IT" in filename) - highest priority
            if program_name.lower() in filename:
                score += direct_bonus
            # Abbreviated match (e.g., "BSIT" in filename)
            elif program_name.replace(' ', '').lower() in filename.replace(' ', ''):
                score += abbreviated_bonus
            # Expanded match using JSON config (e.g., "information technology" for "BS IT")
            else:
                # Get expanded form from JSON config
                expanded_form = self._get_program_full_name_from_config(program_name)
                if expanded_form and expanded_form != program_name:
                    # Extract key terms from the description for matching
                    # e.g., "Bachelor of Science in Information Technology" -> "information technology"
                    import re
                    # Remove degree prefixes and extract the main field
                    clean_description = re.sub(r'^(Bachelor of (Arts|Science) (in|Major in)?|AB|BS)\s*', '', expanded_form, flags=re.IGNORECASE)
                    clean_description = clean_description.lower().strip()
                    
                    if clean_description and clean_description in filename.lower():
                        score += expanded_bonus
        
        # REMOVED: Year level matching to prevent false positives
        # The year level matching was causing "2nd" to match "SECONDARY" documents
        # Year filtering should be handled at the curriculum parsing level, not document retrieval
        
        return min(score, 1.0)
    
    def _is_program_list_query(self, query: str) -> bool:
        """
        Comprehensive detection for program list queries including school/cluster specific queries.
        Handles queries like:
        - "what are the programs under sea"
        - "programs in school of engineering"
        - "what clusters are available"
        - "is this program available"
        """
        query_lower = query.lower().strip()
        
        # Normalize school abbreviations first for better detection
        normalized_query = self._normalize_school_abbreviations(query_lower)
        
        import re
        
        # Pattern 1: General program list queries
        general_patterns = [
            r'\b(what|show|list|all)\s+(programs?|degrees?)\b',
            r'\blist\s+(of\s+)?(all\s+)?programs?\b',
            r'\bprograms?\s+(available|offered)\b',
            r'\b(available|offered)\s+programs?\b',
            r'\bwhat\s+clusters?\b',
            r'\blist\s+clusters?\b'
        ]
        
        # Pattern 2: School-specific queries
        school_patterns = [
            r'\bprograms?\s+(in|under|at|offered\s+by)\s+',
            r'\b(what|show|list)\s+(programs?|degrees?)\s+(are\s+)?(available|offered|in|under)\s+',
            r'\bwhat\s+(are\s+the\s+)?programs?\s+(in|under|at)\s+',
            r'\bprograms?\s+(under|in)\s+\w+',
            r'\b(school|college)\s+of\s+\w+.*programs?\b',
            r'\bsea\s+programs?\b',  # Direct abbreviation
            r'\bsas\s+programs?\b',
            r'\bsbg\s+programs?\b',
            r'\bsoe\s+programs?\b',
            r'\bson\s+programs?\b'
        ]
        
        # Pattern 3: Cluster-specific queries  
        cluster_patterns = [
            r'\bprograms?\s+(in|under)\s+(cluster|humanities|sciences|computer|business|engineering)\b',
            r'\b(cluster|humanities|sciences|computer|business|engineering)\s+programs?\b',
            r'\bwhat\s+(programs?|degrees?)\s+(are\s+in\s+)?(cluster|humanities|sciences|computer|business|engineering)\b'
        ]
        
        # Pattern 4: Program availability queries
        availability_patterns = [
            r'\bis\s+\w+\s+(program\s+)?(available|offered)\b',
            r'\bis\s+\w+\s+\w+\s+(available|offered)\b',  # "is computer science available"
            r'\bis\s+bs\s+\w+\s+(available|offered)\b',   # "is bs it offered"
            r'\bdoes\s+addu\s+(have|offer)\s+\w+\b',
            r'\b(available|offered)\s+at\s+addu\b',
            r'\bprogram\s+(availability|available)\b'
        ]
        
        # Check all patterns
        all_patterns = general_patterns + school_patterns + cluster_patterns + availability_patterns
        
        for pattern in all_patterns:
            if re.search(pattern, query_lower) or re.search(pattern, normalized_query):
                print(f"📋 Program list query detected with pattern: {pattern}")
                return True
        
        # Additional keyword-based detection
        program_list_keywords = [
            'what programs', 'list programs', 'programs available', 'programs offered',
            'programs under', 'programs in', 'programs at', 'show programs',
            'all programs', 'available programs', 'offered programs',
            'what clusters', 'list clusters', 'cluster programs',
            'school programs', 'college programs', 'university programs',
            'program availability', 'is available', 'does addu have',
            'does addu offer', 'programs does addu'
        ]
        
        for keyword in program_list_keywords:
            if keyword in query_lower or keyword in normalized_query:
                print(f"📋 Program list query detected with keyword: {keyword}")
                return True
        
        # Context-aware detection for follow-up queries
        # Check if this looks like a context-enhanced follow-up query about programs
        context_enhanced_patterns = [
            r'\bprograms\s+list\s+what\s+about\b',  # "programs list what about SEA"
            r'\bschool\s+programs\s+what\s+about\b',  # "school programs what about SEA"
            r'\bprograms\s+under\s+what\s+about\b',   # "programs under what about SEA"
            r'\bprograms\s+.*what\s+about\s+\w+\b',   # "programs ... what about [school]"
            r'\blist\s+.*what\s+about\s+\w+\b',       # "list ... what about [school]"
        ]
        
        for pattern in context_enhanced_patterns:
            if re.search(pattern, query_lower) or re.search(pattern, normalized_query):
                print(f"📋 Context-enhanced program list query detected with pattern: {pattern}")
                return True
        
        return False
    
    def _find_program_filename_match(self, program_name_lower: str, filename_lower: str) -> bool:
        """
        Find program matches in filename using normalization config.
        This replaces the hardcoded expansion map with dynamic config-based matching.
        """
        try:
            # Load normalization config
            config = self._load_normalization_config()
            abbreviations = config.get("program_abbreviations", {})
            
            # Search through all schools and programs
            for school_name, school_data in abbreviations.items():
                if isinstance(school_data, dict):  # Skip non-dict entries like "_comment"
                    for abbrev, program_data in school_data.items():
                        # Get the full program name from config
                        full_name = program_data.get("full_name", "").lower()
                        
                        # Check if this matches our target program
                        if full_name == program_name_lower:
                            # Found matching program, now check various forms in filename
                            
                            # 1. Check abbreviation itself (e.g., "bsit" in filename)
                            # Use word boundaries for short abbreviations to avoid false matches
                            import re
                            if len(abbrev) <= 2:
                                # For short abbreviations like "it", use word boundaries
                                pattern = r'\b' + re.escape(abbrev.lower()) + r'\b'
                                if re.search(pattern, filename_lower):
                                    return True
                            else:
                                # For longer abbreviations, simple substring match is fine
                                if abbrev.lower() in filename_lower:
                                    return True
                            
                            # 2. Check description field for expanded forms
                            description = program_data.get("description", "").lower()
                            if description:
                                # Extract key terms from description
                                # e.g., "Bachelor of Science in Information Technology" -> "information technology"
                                if "bachelor of science in" in description:
                                    field_name = description.replace("bachelor of science in", "").strip()
                                    if field_name and field_name in filename_lower:
                                        return True
                                elif "bachelor of arts in" in description:
                                    field_name = description.replace("bachelor of arts in", "").strip()
                                    if field_name and field_name in filename_lower:
                                        return True
                                elif "bachelor of" in description:
                                    field_name = description.replace("bachelor of", "").strip()
                                    if field_name and field_name in filename_lower:
                                        return True
                                
                                # Check if any significant words from description are in filename
                                desc_words = [word for word in description.split() if len(word) > 3 and word not in ['bachelor', 'science', 'arts', 'major']]
                                if len(desc_words) >= 2:
                                    # Check if at least 2 significant words match
                                    matches = sum(1 for word in desc_words if word in filename_lower)
                                    if matches >= 2:
                                        return True
                            
                            # 3. Check all related abbreviations for this program from config
                            # Look for other abbreviations that map to the same full_name
                            for other_abbrev, other_data in school_data.items():
                                if other_data.get("full_name", "").lower() == full_name:
                                    # Check if this abbreviation or its description matches filename
                                    # Use word boundaries for short abbreviations
                                    if len(other_abbrev) <= 2:
                                        pattern = r'\b' + re.escape(other_abbrev.lower()) + r'\b'
                                        if re.search(pattern, filename_lower):
                                            return True
                                    else:
                                        if other_abbrev.lower() in filename_lower:
                                            return True
                                    
                                    other_desc = other_data.get("description", "").lower()
                                    if other_desc:
                                        # Extract field names from descriptions
                                        field_terms = []
                                        
                                        # Common patterns to extract field names
                                        if " abbreviation" in other_desc:
                                            field_name = other_desc.replace(" abbreviation", "").strip()
                                            field_terms.append(field_name)
                                        elif " field name" in other_desc:
                                            field_name = other_desc.replace(" field name", "").strip()
                                            field_terms.append(field_name)
                                        elif " alternate" in other_desc:
                                            field_name = other_desc.replace(" alternate", "").strip()
                                            field_terms.append(field_name)
                                        
                                        # Check extracted terms
                                        for term in field_terms:
                                            if term and len(term) > 2 and term in filename_lower:
                                                return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] Error in _find_program_filename_match: {e}")
            return False
    
    def _calculate_fees_filename_score(self, filename: str, fee_info: dict) -> float:
        """Calculate filename score for fees documents with enhanced program matching"""
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
        
        # ENHANCED: Add program name bonuses (similar to programs but with moderate weights for fees)
        if fee_info.get('program_name'):
            program_name = fee_info['program_name']
            program_name_lower = program_name.lower()
            filename_lower = filename.lower()
            
            # Direct match (e.g., "BS IT" in filename) - moderate bonus for fees
            if program_name_lower in filename_lower:
                score += 0.4
                print(f"📊 Fees direct program match: '{program_name}' in '{filename}'")
            # Abbreviated match (e.g., "BSIT" in filename)
            elif program_name.replace(' ', '').lower() in filename_lower.replace(' ', ''):
                score += 0.3
                print(f"📊 Fees abbreviated program match: '{program_name}' in '{filename}'")
            # Expanded match using JSON config (e.g., "information technology" for "BS IT")
            else:
                # Get expanded form from JSON config
                expanded_form = self._get_program_full_name_from_config(program_name)
                if expanded_form and expanded_form != program_name:
                    # Extract key terms from the description for matching
                    import re
                    # Remove degree prefixes and extract the main field
                    clean_description = re.sub(r'^(Bachelor of (Arts|Science) (in|Major in)?|AB|BS)\s*', '', expanded_form, flags=re.IGNORECASE)
                    clean_description = clean_description.lower().strip()
                    
                    if clean_description and clean_description in filename_lower:
                        score += 0.25
                        print(f"📊 Fees expanded program match: '{clean_description}' in '{filename}'")
        
        return min(score, 1.0)
    
    def _calculate_keyword_score(self, doc_keywords: str, query_lower: str) -> float:
        """Calculate keyword matching score (shared across specializations)"""
        if not doc_keywords or not query_lower:
            return 0.0
        
        query_terms = set(query_lower.split())
        keyword_terms = set(doc_keywords.split())
        
        if not query_terms:
            return 0.0
        
        # IMPROVED: Filter out problematic partial matches
        # Remove year-related terms that can cause false positives
        problematic_terms = {'2nd', '3rd', '4th', 'first', 'second', 'third', 'fourth'}
        
        # Only count exact matches, not partial matches for problematic terms
        valid_matches = 0
        for query_term in query_terms:
            if query_term in problematic_terms:
                # For problematic terms, require exact match
                if query_term in keyword_terms:
                    valid_matches += 1
            else:
                # For other terms, allow partial matching
                if any(query_term in keyword_term for keyword_term in keyword_terms):
                    valid_matches += 1
        
        return valid_matches / len(query_terms)
    
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
    
    def _calculate_programs_content_score(self, content: str, query_lower: str, program_info: dict, query_intent: str = None) -> float:
        """Calculate content score for programs documents"""
        score = 0.0
        query_terms = query_lower.split()
        
        # IMPROVED: Smarter term matching to avoid false positives
        problematic_terms = {'2nd', '3rd', '4th', 'first', 'second', 'third', 'fourth'}
        
        valid_matches = 0
        for term in query_terms:
            if term in problematic_terms:
                # For year-related terms, be more strict to avoid "2nd" matching "secondary"
                # Only count as match if it appears as a standalone word
                import re
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, content, re.IGNORECASE):
                    valid_matches += 1
            else:
                # For other terms, use regular substring matching
                if term in content:
                    valid_matches += 1
        
        if query_terms:
            score += (valid_matches / len(query_terms)) * 0.4
        
        # CONDITIONAL PROGRAM NAME BONUS: Reduce for subject mapping to be more flexible like fees
        if program_info['program_name']:
            program_name = program_info['program_name']
            
            # Adjust bonus based on query intent and specificity
            if query_intent == 'subject_mapping':
                # Check if it's a program-specific subject query vs general subject query
                if program_info.get('program_name'):
                    # Program-specific subject mapping - use STRICT matching (like curriculum)
                    exact_bonus = 0.6  # Restore full bonus for program-specific queries
                    acronym_bonus = 0.5
                    print(f"📊 Using strict content bonuses for program-specific subject mapping")
                else:
                    # General subject mapping - use FLEXIBLE matching (like fees)
                    exact_bonus = 0.3  # Keep reduced bonus for general queries
                    acronym_bonus = 0.25
                    print(f"📊 Using flexible content bonuses for general subject mapping")
            else:
                # Full bonuses for curriculum and other queries
                exact_bonus = 0.6
                acronym_bonus = 0.5
            
            # Check for exact program name match
            if program_name.lower() in content.lower():
                score += exact_bonus
            # Also check for program acronym in content (e.g., "BS ENVISCI")
            elif program_name.replace(' ', '').lower() in content.lower():
                score += acronym_bonus
        
        return min(score, 1.0)
    
    def _calculate_fees_content_score(self, content: str, query_lower: str, fee_info: dict, query_category: str = 'program_specific') -> float:
        """Calculate content score for fees documents with conditional program matching based on query category"""
        score = 0.0
        query_terms = query_lower.split()
        
        # Basic term matching
        matches = sum(1 for term in query_terms if term in content)
        if query_terms:
            score += (matches / len(query_terms)) * 0.3  # Reduced from 0.4 to make room for program bonus
        
        # Bonus for fee type in content
        if fee_info['fee_type'] and fee_info['fee_type'] in content:
            score += 0.2  # Reduced from 0.3 to make room for program bonus
        
        # CONDITIONAL: Only apply program matching bonuses for program-specific queries
        if query_category == 'program_specific' and fee_info.get('program_name'):
            program_name = fee_info['program_name']
            content_lower = content.lower()
            
            # Direct match (e.g., "BS A" in content)
            if program_name.lower() in content_lower:
                score += 0.5
                print(f"📊 Fees direct program match: '{program_name}' in content")
            
            # Expanded form match using JSON config (e.g., "Bachelor of Science in Accountancy")
            else:
                expanded_form = self._get_program_full_name_from_config(program_name)
                if expanded_form and expanded_form.lower() in content_lower:
                    score += 0.5
                    print(f"📊 Fees expanded program match: '{expanded_form}' in content")
                
                # Abbreviated form match (e.g., "BSA" for "BS A")
                elif len(program_name.split()) > 1:  # Only for multi-word programs
                    abbreviated = program_name.replace(' ', '').lower()
                    if abbreviated in content_lower:
                        score += 0.4
                        print(f"📊 Fees abbreviated program match: '{abbreviated}' in content")
        elif query_category in ['policy', 'administrative']:
            print(f"🏛️ Skipping program matching for {query_category} query")
        
        # Bonus for amount-related content if amount was mentioned in query
        if fee_info['amount_mentioned'] and any(term in content for term in ['php', 'peso', 'amount', 'cost']):
            score += 0.2
        
        return min(score, 1.0)

    # ===== SPECIALIZED TOPIC RETRIEVAL FUNCTIONS =====
    
    def retrieve_admissions_documents(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        Specialized retrieval for admissions, enrollment, requirements, and documents.
        Optimized for detecting student types and requirement-specific queries.
        """
        import re
        
        print(f"🎓 Admissions-specialized retrieval for: '{query}'")
        
        try:
            collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)
            
            # Get strategy configuration
            strategy_config = get_retrieval_strategy_config('admissions_specialized')
            document_types = strategy_config.get('document_types', ['admission', 'enrollment', 'scholarship'])
            priorities = strategy_config.get('metadata_priorities', {})
            
            # Detect student type from query
            student_type = self._detect_student_type(query)
            print(f"📝 Detected student type: {student_type}")
            
            # Detect requirement type
            requirement_type = self._detect_requirement_type(query)
            print(f"📋 Detected requirement type: {requirement_type}")
            
            # Get documents with admissions-specific filtering
            where_clause = {
                "$and": [
                    {"source": "pdf_scrape"},
                    {"$or": [{"document_type": doc_type} for doc_type in document_types]}
                ]
            }
            
            all_docs = collection.get(
                where=where_clause,
                include=["documents", "metadatas"]
            )
            
            all_ids = all_docs.get('ids', [])
            all_contents = all_docs.get('documents', [])
            all_metadatas = all_docs.get('metadatas', [])
            
            print(f"📚 Found {len(all_ids)} admissions-related documents")
            
            # Score documents with admissions-specific logic
            scored_results = []
            query_lower = query.lower()
            
            for i, (doc_id, content, metadata) in enumerate(zip(all_ids, all_contents, all_metadatas)):
                filename = metadata.get('filename', '').lower()
                doc_keywords = metadata.get('keywords', '').lower()
                content_lower = content.lower()
                
                # Calculate specialized scores
                filename_score = self._calculate_admissions_filename_score(filename, student_type, requirement_type)
                keyword_score = self._calculate_keyword_score(doc_keywords, query_lower)
                content_score = self._calculate_admissions_content_score(content_lower, query_lower, student_type)
                
                # Apply strategy priorities
                final_score = (
                    filename_score * priorities.get('filename', 0.4) +
                    keyword_score * priorities.get('keywords', 0.3) +
                    content_score * priorities.get('content', 0.3)
                )
                
                if final_score > 0:
                    scored_results.append({
                        'id': doc_id,
                        'content': content,
                        'relevance': final_score,
                        'folder': metadata.get('folder_name', 'Unknown'),
                        'document_type': metadata.get('document_type', 'other'),
                        'target_program': metadata.get('target_program', 'all'),
                        'filename': metadata.get('filename', ''),
                        'retrieval_strategy': 'admissions_specialized',
                        'current_topic': 'admissions_enrollment',
                        '_debug': {
                            'filename_score': filename_score,
                            'keyword_score': keyword_score,
                            'content_score': content_score,
                            'student_type': student_type,
                            'requirement_type': requirement_type
                        }
                    })
            
            # Sort and return top results
            scored_results.sort(key=lambda x: x['relevance'], reverse=True)
            
            print(f"✅ Top {min(top_k, len(scored_results))} admissions results:")
            for i, doc in enumerate(scored_results[:top_k]):
                debug = doc['_debug']
                print(f"   {i+1}. {doc['filename'][:60]}")
                print(f"       Score: {doc['relevance']:.3f} (f={debug['filename_score']:.2f}, k={debug['keyword_score']:.2f}, c={debug['content_score']:.2f})")
                print(f"       Student type: {debug['student_type']}, Requirement: {debug['requirement_type']}")
            
            return scored_results[:top_k]
            
        except Exception as e:
            print(f"❌ Admissions retrieval error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def retrieve_programs_documents(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        Simplified retrieval for programs, courses, and curriculum.
        Uses the same flexible approach as fees: get all documents, then filter by keywords OR document_type.
        """
        import re
        
        print(f"📚 Programs-simplified retrieval for: '{query}'")
        
        try:
            collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)
            
            # Get strategy configuration
            strategy_config = get_retrieval_strategy_config('programs_specialized')
            document_types = strategy_config.get('document_types', ['academic', 'curriculum'])
            priorities = strategy_config.get('metadata_priorities', {})
            
            # Extract program information from query
            program_info = self._extract_program_info(query)
            print(f"🎯 Extracted program info: {program_info}")
            
            # Step 1: Get documents with programs-specific document_type filtering
            where_clause = {
                "$and": [
                    {"source": "pdf_scrape"},
                    {"$or": [{"document_type": doc_type} for doc_type in document_types]}
                ]
            }
            
            all_docs = collection.get(
                where=where_clause,
                include=["documents", "metadatas"]
            )
            
            all_ids = all_docs.get('ids', [])
            all_contents = all_docs.get('documents', [])
            all_metadatas = all_docs.get('metadatas', [])
            
            # Step 2: Also get ALL documents that might have program keywords but not the right document_type
            # This is to catch documents that might not have been stored with correct document_type
            all_docs_by_keywords = collection.get(
                where={"source": "pdf_scrape"},
                include=["documents", "metadatas"]
            )
            
            # Step 3: Combine and deduplicate documents
            all_combined_ids = list(set(all_ids + all_docs_by_keywords.get('ids', [])))
            
            # Get full data for combined documents
            if len(all_combined_ids) > len(all_ids):
                print(f"📚 Expanded search to {len(all_combined_ids)} total documents")
                combined_docs = collection.get(
                    ids=all_combined_ids,
                    include=["documents", "metadatas"]
                )
                all_ids = combined_docs.get('ids', [])
                all_contents = combined_docs.get('documents', [])
                all_metadatas = combined_docs.get('metadatas', [])
            
            # Get topic keywords for additional filtering
            from .topics import get_topic_keywords
            topic_keywords = get_topic_keywords('programs_courses')
            print(f"📝 Using topic keywords for additional filtering: {topic_keywords}")
            
            # Step 4: Filter documents: must have program keywords OR correct document_type
            filtered_docs = []
            for i, (doc_id, content, metadata) in enumerate(zip(all_ids, all_contents, all_metadatas)):
                doc_keywords = metadata.get('keywords', '').lower()
                filename = metadata.get('filename', '').lower()
                
                # Check if document has program-related keywords
                has_program_keywords = False
                if topic_keywords:
                    for keyword in topic_keywords:
                        if keyword.lower() in doc_keywords or keyword.lower() in filename:
                            has_program_keywords = True
                            break
                
                # Include if it has programs document_type OR program keywords
                doc_type = metadata.get('document_type', '')
                if doc_type in document_types or has_program_keywords:
                    filtered_docs.append((doc_id, content, metadata))
                    if has_program_keywords and doc_type not in document_types:
                        print(f"✅ Including document with program keywords: {metadata.get('filename', 'N/A')}")
            
            print(f"✅ After filtering: {len(filtered_docs)} documents")
            
            if not filtered_docs:
                print(f"❌ No program documents found after filtering")
                return []
            
            # Score documents with programs-specific logic
            scored_results = []
            query_lower = query.lower()
            
            for i, (doc_id, content, metadata) in enumerate(filtered_docs):
                filename = metadata.get('filename', '').lower()
                doc_keywords = metadata.get('keywords', '').lower()
                content_lower = content.lower()
                
                # Calculate specialized scores
                filename_score = self._calculate_programs_filename_score(filename, program_info)
                keyword_score = self._calculate_keyword_score(doc_keywords, query_lower)
                content_score = self._calculate_programs_content_score(content_lower, query_lower, program_info)
                
                # Apply strategy priorities
                final_score = (
                    filename_score * priorities.get('filename', 0.5) +
                    keyword_score * priorities.get('keywords', 0.3) +
                    content_score * priorities.get('content', 0.2)
                )
                
                if final_score > 0:
                    scored_results.append({
                        'id': doc_id,
                        'content': content,
                        'relevance': final_score,
                        'folder': metadata.get('folder_name', 'Unknown'),
                        'document_type': metadata.get('document_type', 'other'),
                        'target_program': metadata.get('target_program', 'all'),
                        'filename': metadata.get('filename', ''),
                        'retrieval_strategy': 'programs_specialized',
                        'current_topic': 'programs_courses',
                        '_debug': {
                            'filename_score': filename_score,
                            'keyword_score': keyword_score,
                            'content_score': content_score,
                            'program_info': program_info
                        }
                    })
            
            # Sort and return top results
            scored_results.sort(key=lambda x: x['relevance'], reverse=True)
            
            print(f"✅ Top {min(top_k, len(scored_results))} programs results:")
            for i, doc in enumerate(scored_results[:top_k]):
                debug = doc['_debug']
                print(f"   {i+1}. {doc['filename'][:60]}")
                print(f"       Score: {doc['relevance']:.3f} (f={debug['filename_score']:.2f}, k={debug['keyword_score']:.2f}, c={debug['content_score']:.2f})")
                print(f"       Program info: {debug['program_info']}")
            
            return scored_results[:top_k]
            
        except Exception as e:
            print(f"❌ Programs retrieval error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def retrieve_fees_documents(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        Specialized retrieval for fees, payments, and financial information.
        Includes all documents with fees or financial document types.
        """
        import re
        
        print(f"💰 Fees-specialized retrieval for: '{query}'")
        
        try:
            collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)
            
            # Get strategy configuration
            strategy_config = get_retrieval_strategy_config('fees_specialized')
            document_types = strategy_config.get('document_types', ['fees', 'financial'])
            priorities = strategy_config.get('metadata_priorities', {})
            
            # Extract fee-related information from query
            fee_info = self._extract_fee_info(query)
            print(f"💳 Extracted fee info: {fee_info}")
            
            # Get documents with fees-specific filtering (same pattern as admissions/programs)
            where_clause = {
                "$and": [
                    {"source": "pdf_scrape"},
                    {"$or": [{"document_type": doc_type} for doc_type in document_types]}
                ]
            }
            
            all_docs = collection.get(
                where=where_clause,
                include=["documents", "metadatas"]
            )
            
            all_ids = all_docs.get('ids', [])
            all_contents = all_docs.get('documents', [])
            all_metadatas = all_docs.get('metadatas', [])
            
            print(f"📚 Found {len(all_ids)} documents with fees document_type")
            
            # Also get documents that might have fee keywords but not the right document_type
            # This is to catch CSV files that might not have been stored with correct document_type
            all_docs_by_keywords = collection.get(
                where={"source": "pdf_scrape"},
                include=["documents", "metadatas"]
            )
            
            # Combine and deduplicate documents
            all_combined_ids = list(set(all_ids + all_docs_by_keywords.get('ids', [])))
            
            # Get full data for combined documents
            if len(all_combined_ids) > len(all_ids):
                print(f"📚 Expanded search to {len(all_combined_ids)} total documents")
                combined_docs = collection.get(
                    ids=all_combined_ids,
                    include=["documents", "metadatas"]
                )
                all_ids = combined_docs.get('ids', [])
                all_contents = combined_docs.get('documents', [])
                all_metadatas = combined_docs.get('metadatas', [])
            
            # Get topic keywords for additional filtering
            from .topics import get_topic_keywords
            topic_keywords = get_topic_keywords('fees')
            print(f"📝 Using topic keywords for additional filtering: {topic_keywords}")
            
            # Filter documents: must have fee keywords
            filtered_docs = []
            for i, (doc_id, content, metadata) in enumerate(zip(all_ids, all_contents, all_metadatas)):
                doc_keywords = metadata.get('keywords', '').lower()
                filename = metadata.get('filename', '').lower()
                
                
                # Check if document has fee-related keywords or is already from document_type filtering
                has_fee_keywords = False
                if topic_keywords:
                    for keyword in topic_keywords:
                        if keyword.lower() in doc_keywords or keyword.lower() in filename:
                            has_fee_keywords = True
                            break
                
                # Include if it has fees document_type OR fee keywords
                doc_type = metadata.get('document_type', '')
                if doc_type in ['fees', 'financial'] or has_fee_keywords:
                    filtered_docs.append((doc_id, content, metadata))
                    if has_fee_keywords and doc_type not in ['fees', 'financial']:
                        print(f"✅ Including document with fee keywords: {metadata.get('filename', 'N/A')}")
            
            print(f"✅ After filtering: {len(filtered_docs)} documents")
            
            if not filtered_docs:
                print(f"❌ No fee documents found after filtering")
                return []
            
            # Score documents with fees-specific logic
            scored_results = []
            query_lower = query.lower()
            
            for doc_id, content, metadata in filtered_docs:
                filename = metadata.get('filename', '').lower()
                doc_keywords = metadata.get('keywords', '').lower()
                content_lower = content.lower()
                
                # Calculate specialized scores
                filename_score = self._calculate_fees_filename_score(filename, fee_info)
                keyword_score = self._calculate_keyword_score(doc_keywords, query_lower)
                content_score = self._calculate_fees_content_score(content_lower, query_lower, fee_info)
                
                # Apply strategy priorities
                final_score = (
                    filename_score * priorities.get('filename', 0.3) +
                    keyword_score * priorities.get('keywords', 0.4) +
                    content_score * priorities.get('content', 0.3)
                )
                
                if final_score > 0:
                    scored_results.append({
                        'id': doc_id,
                        'content': content,
                        'relevance': final_score,
                        'folder': metadata.get('folder_name', 'Unknown'),
                        'document_type': metadata.get('document_type', 'other'),
                        'target_program': metadata.get('target_program', 'all'),
                        'filename': metadata.get('filename', ''),
                        'retrieval_strategy': 'fees_specialized',
                        'current_topic': 'fees',
                        '_debug': {
                            'filename_score': filename_score,
                            'keyword_score': keyword_score,
                            'content_score': content_score,
                            'fee_info': fee_info
                        }
                    })
            
            # Sort and return top results
            scored_results.sort(key=lambda x: x['relevance'], reverse=True)
            
            print(f"✅ Top {min(top_k, len(scored_results))} fees results:")
            for i, doc in enumerate(scored_results[:top_k]):
                debug = doc['_debug']
                print(f"   {i+1}. {doc['filename'][:60]}")
                print(f"       Score: {doc['relevance']:.3f} (f={debug['filename_score']:.2f}, k={debug['keyword_score']:.2f}, c={debug['content_score']:.2f})")
                print(f"       Fee info: {debug['fee_info']}")
            
            return scored_results[:top_k]
            
        except Exception as e:
            print(f"❌ Fees retrieval error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def retrieve_programs_by_subject(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        NEW METHOD: Retrieve programs that contain specific subjects
        Handles various query types:
        - "what programs have math" (general positive)
        - "what programs don't offer OJT" (general negative)
        - "what math subjects are in BS CS" (program-specific)
        - "how many programming courses in BS IT" (count)
        """
        print(f"🔍 Subject-to-program mapping for: '{query}'")
        
        # Extract subject and query type information
        subject_info = self._extract_subject_from_query(query)
        
        # Get query type (general, negative, program_specific, count)
        query_type = subject_info.get('query_type', 'general')
        print(f"📝 Extracted subject: '{subject_info['subject']}'")
        print(f"🔍 Query type: {query_type}")
        
        # For backward compatibility, also check the old way
        is_negative_query = 'negative' in query_type or self._is_negative_subject_query(query)
        cluster_filter = self._detect_cluster_query(query)
        
        print(f"🔄 Negative query: {is_negative_query}")
        if cluster_filter:
            print(f"🎯 Cluster filter: {cluster_filter}")
        
        # Get ALL curriculum documents
        all_curriculum_docs = self._get_all_curriculum_documents()
        
        # Apply cluster filtering using JSON config
        if cluster_filter:
            print(f"🎯 Applying cluster filter: {cluster_filter}")
            curriculum_docs = self._apply_cluster_filter_to_curriculum_docs(all_curriculum_docs, cluster_filter)
            print(f"📚 Filtered to {len(curriculum_docs)} documents in {cluster_filter} cluster")
        else:
            curriculum_docs = all_curriculum_docs
        
        # Parse subjects from each curriculum document
        program_subject_map = {}
        for doc in curriculum_docs:
            program_acronym = self._extract_program_acronym_from_curriculum(doc['content'])
            if program_acronym:
                subjects = self._extract_subjects_from_curriculum_content(doc['content'])
                program_subject_map[program_acronym] = subjects
        
        print(f"📊 Parsed {len(program_subject_map)} programs with subjects")
        
        # For program-specific or count queries, filter to just that program
        target_program = subject_info.get('program')
        if target_program and (query_type in ['program_specific', 'negative_program_specific', 'count']):
            print(f"🎯 Filtering to specific program: {target_program}")
            # Normalize the program name for better matching
            normalized_program = target_program.upper().strip()
            
            # Filter the program_subject_map
            filtered_map = {}
            for program, subjects in program_subject_map.items():
                # Simple program name matching (could be enhanced)
                if normalized_program in program.upper() or program.upper() in normalized_program:
                    filtered_map[program] = subjects
                    print(f"✅ Found matching program: {program}")
            
            # Update the map to only include the target program
            if filtered_map:
                program_subject_map = filtered_map
            else:
                print(f"⚠️ No matching program found for: {target_program}")
        
        # Find matching programs using semantic matching
        matching_programs = self._find_programs_with_subject_semantic(
            program_subject_map, 
            subject_info, 
            is_negative_query
        )
        
        # Add query type to subject_info for response formatting
        subject_info['query_type'] = query_type
        
        # Organize matching programs by year and semester for better LLM formatting
        organized_programs = {}
        for program, subjects in matching_programs.items():
            # Group subjects by year and semester
            year_semester_map = {}
            for subject in subjects:
                year = subject.get('year', 'Unknown')
                semester = subject.get('semester') or 'unknown'  # Handle None values
                
                if year not in year_semester_map:
                    year_semester_map[year] = {}
                if semester not in year_semester_map[year]:
                    year_semester_map[year][semester] = []
                
                year_semester_map[year][semester].append(subject)
            
            organized_programs[program] = {
                'subjects_by_year_semester': year_semester_map,
                'total_subjects': len(subjects),
                'all_subjects': subjects  # Keep original flat list for backward compatibility
            }
        
        print(f"📊 Organized {len(organized_programs)} programs with year/semester structure")
        
        # Return enhanced data with year/semester organization for better LLM formatting
        return [{
            'id': 'subject_mapping_organized_data',
            'matching_programs': organized_programs,
            'subject_info': subject_info,
            'is_negative': is_negative_query,
            'cluster_filter': cluster_filter,
            'total_programs': len(organized_programs),
            'has_year_semester_context': True
        }]

    def _extract_subject_from_query(self, query: str) -> dict:
        """Extract the target subject from the user query with enhanced query type detection"""
        import re
        
        query_lower = query.lower()
        
        # Check for count queries first (how many X subjects in Y)
        count_match = re.search(r'how many (\w+) (?:subjects|courses|classes) (?:in|does) ([^?]+)', query_lower)
        if count_match:
            subject = count_match.group(1).strip()
            program = count_match.group(2).strip()
            return {
                'subject': subject,
                'program': program,
                'query_type': 'count',
                'original_query': query
            }
            
        # Check for program-specific subject queries (what X subjects are in Y)
        program_specific_match = re.search(r'what (?:are )?(?:the )?(\w+) (?:subjects|courses|classes) (?:are )?(?:in|under|for) ([^?]+)', query_lower)
        if program_specific_match:
            subject = program_specific_match.group(1).strip()
            program = program_specific_match.group(2).strip()
            return {
                'subject': subject,
                'program': program,
                'query_type': 'program_specific',
                'original_query': query
            }
        
        # Enhanced patterns to extract subject for general queries
        subject_patterns = [
            # Positive patterns
            r'programs?\s+(?:have|offer|include|contain|teach)\s+([^?]+)',
            r'programs?\s+(?:with|that\s+have|that\s+offer|that\s+include)\s+([^?]+)',
            r'(?:what|which)\s+programs?\s+(?:have|offer|include|contain|teach)\s+([^?]+)',
            
            # Enhanced negative patterns
            r'programs?\s+(?:do\s+not|don\'t|doesn\'t|doesnt)\s+(?:have|offer|include|provide)\s+([^?]+)',
            r'programs?\s+(?:without|that\s+do\s+not|that\s+don\'t|that\s+doesn\'t)\s+([^?]+)',
            r'(?:what|which)\s+programs?\s+(?:do\s+not|don\'t|doesn\'t)\s+(?:have|offer|include|provide)\s+([^?]+)',
            r'programs?\s+that\s+(?:doesn\'t|doesnt)\s+(?:offer|have|include|provide)\s+([^?]+)',
            
            # Subject-in-program patterns
            r'(?:subjects?|courses?)\s+(?:in|under|for)\s+([^?]+)',
        ]
        
        # Check for program context in subject queries (X in program Y)
        program_context_pattern = r'([^?]+)\s+(?:in|under|for)\s+([^?]+)'
        
        for pattern in subject_patterns:
            match = re.search(pattern, query_lower)
            if match:
                subject = match.group(1).strip()
                # Clean up common words and punctuation
                subject = re.sub(r'\b(the|a|an|in|on|at|for|of|with)\b', '', subject).strip()
                # Remove trailing punctuation (question marks, periods, etc.)
                subject = re.sub(r'[?.,!;]+$', '', subject).strip()
                
                # Check if there's a program context
                program_match = re.search(program_context_pattern, subject)
                if program_match:
                    actual_subject = program_match.group(1).strip()
                    program = program_match.group(2).strip()
                    # Clean up punctuation from both subject and program
                    actual_subject = re.sub(r'[?.,!;]+$', '', actual_subject).strip()
                    program = re.sub(r'[?.,!;]+$', '', program).strip()
                    # Determine if this is a negative query
                    query_type = 'program_specific'
                    negative_indicators = ['without', 'not have', 'don\'t have', 'do not have', 'not offer', 'don\'t offer', 
                                         'doesn\'t have', 'doesn\'t offer', 'doesnt have', 'doesnt offer']
                    if any(neg in query_lower for neg in negative_indicators):
                        query_type = 'negative_program_specific'
                        print(f"📝 Detected negative program-specific query: {query}")
                    
                    return {
                        'subject': actual_subject, 
                        'program': program,
                        'query_type': query_type,
                        'original_query': query
                    }
                
                # Determine if this is a negative query with enhanced detection
                query_type = 'general'
                negative_indicators = ['without', 'not have', 'don\'t have', 'do not have', 'not offer', 'don\'t offer',
                                     'doesn\'t have', 'doesn\'t offer', 'doesnt have', 'doesnt offer', 'that do not',
                                     'that don\'t', 'that doesn\'t', 'that doesnt']
                if any(neg in query_lower for neg in negative_indicators):
                    query_type = 'negative'
                    print(f"📝 Detected negative general query: {query}")
                else:
                    print(f"📝 Detected positive general query: {query}")
                
                return {
                    'subject': subject, 
                    'query_type': query_type,
                    'original_query': query
                }
        
        # Fallback: extract last meaningful word
        words = query_lower.split()
        meaningful_words = [w for w in words if len(w) > 2 and w not in ['what', 'which', 'programs', 'have', 'offer', 'the', 'and', 'or']]
        if meaningful_words:
            return {
                'subject': meaningful_words[-1], 
                'query_type': 'general',
                'original_query': query
            }
        
        return {
            'subject': '', 
            'query_type': 'unknown',
            'original_query': query
        }

    def _extract_subject_type_from_query(self, query: str) -> str:
        """Extract subject type from query with enhanced specificity"""
        query_lower = query.lower()
        
        # More specific subject type patterns to avoid conflicts
        subject_patterns = {
            'programming': ['programming', 'coding', 'computer programming', 'software development'],
            'mathematics': ['math subjects', 'mathematics subjects', 'math courses', 'mathematics courses', 
                           'about math', 'what about math', 'math topics'],  # More specific
            'science': ['science subjects', 'science courses', 'about science', 'what about science'],
            'english': ['english subjects', 'english courses', 'literature', 'writing'],
            'business': ['business subjects', 'business courses', 'management', 'accounting', 'finance'],
            'engineering': ['engineering subjects', 'engineering courses', 'technical', 'design'],
            'general': ['subjects', 'courses', 'curriculum']
        }
        
        # Check specific patterns first, then fall back to general
        for subject_type, patterns in subject_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                print(f"🎯 Detected subject type '{subject_type}' from pattern matching")
                return subject_type
        
        return 'general'

    def _detect_incorrect_program_normalization(self, original_query: str, enhanced_query: str, subject_type: str) -> bool:
        """Detect if a subject word was incorrectly normalized to a program name"""
        
        # Common subject words that get incorrectly normalized
        subject_to_program_mappings = {
            'math': ('BS MATH', 'mathematics'),
            'mathematics': ('BS MATH', 'mathematics'), 
            'bio': ('BS BIO', 'science'),
            'biology': ('BS BIO', 'science'),
            'chem': ('BS CHEM', 'science'),
            'chemistry': ('BS CHEM', 'science')
        }
        
        original_lower = original_query.lower()
        enhanced_lower = enhanced_query.lower()
        
        # Additional check: ensure it's in subject context
        subject_indicators = ['subjects', 'courses', 'about', 'topics']
        has_subject_context = any(indicator in original_lower for indicator in subject_indicators)
        
        if not has_subject_context:
            return False
        
        # Check if a subject word was normalized to its program equivalent
        for subject_word, (program_name, expected_subject_type) in subject_to_program_mappings.items():
            if (subject_word in original_lower and 
                program_name.lower() in enhanced_lower and
                subject_type == expected_subject_type):
                
                print(f"🚨 Detected incorrect normalization: '{subject_word}' → '{program_name}' in subject context")
                return True
        
        return False

    def _restore_subject_context(self, original_query: str, subject_type: str) -> str:
        """Restore original subject context by removing incorrect program normalization"""
        
        # Get session program for proper context
        session_program = self.session_state.get('current_program')
        
        if session_program and subject_type != 'general':
            # Map subject type to query-friendly term
            subject_term_map = {
                'mathematics': 'math',
                'science': 'science',
                'programming': 'programming',
                'english': 'english',
                'business': 'business',
                'engineering': 'engineering'
            }
            
            subject_term = subject_term_map.get(subject_type, subject_type)
            
            # Build proper subject mapping query
            restored_query = f"{session_program} {subject_term} subjects"
            print(f"🔧 Restored query: '{original_query}' → '{restored_query}'")
            return restored_query
        
        return original_query

    def _check_subject_mapping_ambiguity(self, program: str, subject_type: str, query: str) -> dict:
        """Check if subject mapping query is ambiguous and needs clarification"""
        
        # Get recent context history
        subject_state = self.session_state.get('subject_mapping_state', {})
        context_history = subject_state.get('context_history', [])
        
        # Check for vague queries without explicit context
        vague_patterns = ['what about', 'how about', 'what are the', 'show me']
        is_vague_query = any(pattern in query.lower() for pattern in vague_patterns)
        
        if not is_vague_query:
            return {'is_ambiguous': False}
        
        # Check recent context for multiple programs
        recent_programs = set()
        recent_subjects = set()
        current_time = time.time()
        
        for entry in context_history:
            # Consider entries from last 5 minutes as "recent"
            if current_time - entry['timestamp'] < 300:
                if entry['program']:
                    recent_programs.add(entry['program'])
                if entry['subject_type']:
                    recent_subjects.add(entry['subject_type'])
        
        # Also check curriculum state for cross-query context
        curriculum_state = self.session_state.get('curriculum_state', {})
        if curriculum_state.get('current_program'):
            recent_programs.add(curriculum_state['current_program'])
        
        # Determine if clarification is needed
        needs_program_clarification = len(recent_programs) > 1 and not program
        needs_subject_clarification = len(recent_subjects) > 1 and not subject_type
        
        if needs_program_clarification:
            programs_list = ', '.join(recent_programs)
            return {
                'is_ambiguous': True,
                'clarification_message': f"Which program's subjects are you asking about? Recent programs discussed: {programs_list}"
            }
        
        if needs_subject_clarification:
            subjects_list = ', '.join(recent_subjects)
            return {
                'is_ambiguous': True,
                'clarification_message': f"Which type of subjects? Recent types discussed: {subjects_list}"
            }
        
        return {'is_ambiguous': False}

    def _detect_query_subtype(self, query: str) -> dict:
        """
        Detect the specific subtype of subject mapping query and return structured classification.
        
        Returns:
            dict with keys: subtype, is_negative, is_program_specific, is_count, confidence
        """
        import re
        
        query_lower = query.lower()
        
        # Initialize classification
        classification = {
            'subtype': 'general_positive',
            'is_negative': False,
            'is_program_specific': False,
            'is_count': False,
            'confidence': 0.0,
            'matched_patterns': []
        }
        
        # Count query detection
        count_patterns = [
            r'\bhow\s+many\s+\w+\s+(?:subjects?|courses?)\s+(?:in|does)\s+',
            r'\bhow\s+many\s+(?:subjects?|courses?)\s+(?:of|about)\s+\w+\s+(?:in|does)\s+'
        ]
        
        for pattern in count_patterns:
            if re.search(pattern, query_lower):
                classification['subtype'] = 'count'
                classification['is_count'] = True
                classification['is_program_specific'] = True
                classification['confidence'] = 0.9
                classification['matched_patterns'].append(f"Count: {pattern}")
                print(f"🔢 Detected count query with pattern: {pattern}")
                return classification
        
        # Program-specific query detection
        program_specific_patterns = [
            r'\bwhat\s+\w+\s+(?:subjects?|courses?)\s+(?:are\s+)?(?:in|offered\s+in|taught\s+in)\s+',
            r'\blist\s+\w+\s+(?:subjects?|courses?)\s+(?:in|for)\s+',
            r'\b(?:subjects?|courses?)\s+(?:in|offered\s+in|taught\s+in)\s+\w+',
            r'\bwhat\s+(?:subjects?|courses?)\s+(?:does|are\s+offered\s+in)\s+\w+\s+(?:have|offer)\s*'
        ]
        
        for pattern in program_specific_patterns:
            if re.search(pattern, query_lower):
                classification['is_program_specific'] = True
                classification['confidence'] = 0.8
                classification['matched_patterns'].append(f"Program-specific: {pattern}")
                print(f"🎯 Detected program-specific query with pattern: {pattern}")
                break
        
        # Negative query detection
        negative_patterns = [
            r'\bdo\s+not\s+(?:have|offer|include|provide)\b',
            r'\bdon\'t\s+(?:have|offer|include|provide)\b',
            r'\bdoesn\'t\s+(?:have|offer|include|provide)\b',
            r'\bdoesnt\s+(?:have|offer|include|provide)\b',
            r'\bwithout\b',
            r'\bthat\s+(?:do\s+not|don\'t|doesn\'t|doesnt)\s+(?:have|offer|include|provide)\b'
        ]
        
        for pattern in negative_patterns:
            if re.search(pattern, query_lower):
                classification['is_negative'] = True
                classification['confidence'] = max(classification['confidence'], 0.85)
                classification['matched_patterns'].append(f"Negative: {pattern}")
                print(f"❌ Detected negative query with pattern: {pattern}")
                break
        
        # Determine final subtype
        if classification['is_count']:
            classification['subtype'] = 'count'
        elif classification['is_program_specific'] and classification['is_negative']:
            classification['subtype'] = 'program_specific_negative'
        elif classification['is_program_specific']:
            classification['subtype'] = 'program_specific_positive'
        elif classification['is_negative']:
            classification['subtype'] = 'general_negative'
        else:
            classification['subtype'] = 'general_positive'
            classification['confidence'] = max(classification['confidence'], 0.7)
        
        print(f"📊 Query classification: {classification['subtype']} (confidence: {classification['confidence']:.2f})")
        
        return classification

    def _is_negative_subject_query(self, query: str) -> bool:
        """Check if this is a negative query (programs that DON'T have something)"""
        import re
        
        negative_patterns = [
            r'\bdo\s+not\b',
            r'\bdon\'t\b',
            r'\bdoesn\'t\b',
            r'\bdoesnt\b',
            r'\bwithout\b',
            r'\bnot\s+offer\b',
            r'\bnot\s+have\b',
            r'\bnot\s+include\b',
            r'\bnot\s+provide\b',
            r'\bdon\'t\s+offer\b',
            r'\bdon\'t\s+have\b',
            r'\bdon\'t\s+include\b',
            r'\bdon\'t\s+provide\b',
            r'\bdoesn\'t\s+offer\b',
            r'\bdoesn\'t\s+have\b',
            r'\bdoesn\'t\s+include\b',
            r'\bdoesn\'t\s+provide\b',
            r'\bthat\s+do\s+not\s+(offer|have|include|provide)\b',
            r'\bthat\s+don\'t\s+(offer|have|include|provide)\b',
            r'\bthat\s+doesn\'t\s+(offer|have|include|provide)\b'
        ]
        
        query_lower = query.lower()
        matched_negative = False
        for pattern in negative_patterns:
            if re.search(pattern, query_lower):
                matched_negative = True
                print(f"🔍 Negative query detected with pattern: {pattern}")
                break
        
        return matched_negative

    def _get_all_curriculum_documents(self) -> List[Dict]:
        """Get all curriculum documents from ChromaDB"""
        try:
            from .chroma_connection import ChromaService
            collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)
            
            # Get documents with curriculum-related document types
            # Based on analysis: curriculum documents are stored as "academic" type
            where_clause = {
                "$and": [
                    {"source": "pdf_scrape"},
                    {"$or": [
                        {"document_type": "curriculum"},
                        {"document_type": "academic"}  # This is where curriculum docs are actually stored
                    ]}
                ]
            }
            
            all_docs = collection.get(
                where=where_clause,
                include=["documents", "metadatas"]
            )
            
            curriculum_docs = []
            for doc_id, content, metadata in zip(
                all_docs.get('ids', []), 
                all_docs.get('documents', []), 
                all_docs.get('metadatas', [])
            ):
                curriculum_docs.append({
                    'id': doc_id,
                    'content': content,
                    'metadata': metadata
                })
            
            print(f"📚 Retrieved {len(curriculum_docs)} curriculum documents")
            return curriculum_docs
            
        except Exception as e:
            print(f"❌ Error retrieving curriculum documents: {e}")
            return []

    def _extract_program_acronym_from_curriculum(self, content: str) -> str:
        """Extract program acronym from curriculum document content"""
        import re
        
        # Pattern to match "Program Acronym: BS IT" or "Acronym: BS IT"
        acronym_pattern = r'(?:Program\s+)?Acronym:\s*([A-Z][A-Z\s\-]*?)(?:\n|$|Effective|Curriculum)'
        match = re.search(acronym_pattern, content, re.IGNORECASE)
        
        if match:
            acronym = match.group(1).strip()
            # Clean up common suffixes
            acronym = re.sub(r'\s*(Effective|Academic|Year|Curriculum).*$', '', acronym, re.IGNORECASE)
            return acronym.strip()
        
        # Fallback: look for specific program patterns at the beginning
        program_patterns = [
            r'(BS\s+[A-Z]+(?:\s*-\s*[A-Z]+)?)',  # BS IT, BS CS, BS ENTREP - A
            r'(AB\s+[A-Z]+(?:\s*-\s*[A-Z]+)?)',  # AB ENG, AB MC
            r'(BSED\s*-?\s*[A-Z]+)',             # BSED - ENGLISH
            r'(BSN|BPM|BECE|BEED)',              # Single acronyms
        ]
        
        # Look for patterns in the first 500 characters
        content_start = content[:500]
        for pattern in program_patterns:
            matches = re.findall(pattern, content_start, re.IGNORECASE)
            if matches:
                # Return the first match, cleaned up
                acronym = matches[0].strip()
                return acronym
        
        return None

    def _extract_subjects_from_curriculum_content(self, content: str) -> List[Dict]:
        """Extract subjects and course codes from curriculum document content with year/semester context"""
        import re
        
        subjects = []
        
        # Track current year and semester context
        current_year = None
        current_semester = None
        
        # Updated pattern to match "ASF 1102 - NATIONAL SERVICE TRAINING PROGRAM"
        # Allows for spaces in the code and handles various formats
        course_pattern = r'([A-Z]{2,4}\s*\d{2,4})\s*-\s*([A-Z][^\n\r]+?)(?:\s*\d+\.\d+\.\s*(?:Pay|Credit)\s*Units|$)'
        
        # Process content line by line to track year/semester context
        lines = content.split('\n')
        seen_codes = set()  # Avoid duplicates
        
        for line in lines:
            line = line.strip()
            
            # Detect combined year-semester headers (e.g., "I. Year 1 — First Semester (27.0 CU)")
            combined_match = re.search(r'Year\s+(\d+)\s*[—-]\s*(First|Second|Summer)(?:\s+Semester)?\s*\(', line, re.IGNORECASE)
            if combined_match:
                current_year = int(combined_match.group(1))
                current_semester = combined_match.group(2).lower()
                print(f"📅 Found Year {current_year} - {current_semester} semester context")
                continue
            
            # Detect standalone year headers (e.g., "Year 1", "YEAR 3")
            year_match = re.search(r'Year\s+(\d+)', line, re.IGNORECASE)
            if year_match and not combined_match:  # Only if not already matched above
                current_year = int(year_match.group(1))
                current_semester = None  # Reset semester when new year starts
                print(f"📅 Found Year {current_year} context")
                continue
                
            # Detect standalone semester headers (e.g., "First Semester", "Second Semester", "Summer")
            semester_match = re.search(r'^(?:I+\.\s*)?(First|Second|Summer)(?:\s+Semester)?\s*(?:\(|$)', line, re.IGNORECASE)
            if semester_match:
                current_semester = semester_match.group(1).lower()
                print(f"📅 Found {current_semester} semester context (Year {current_year})")
                continue
            
            # Look for course codes in this line
            matches = re.findall(course_pattern, line)
            for code, title in matches:
                # Clean up the code and title
                code = re.sub(r'\s+', ' ', code.strip())  # Normalize spaces
                title = title.strip()
                
                # Skip if we've already seen this code
                if code in seen_codes:
                    continue
                seen_codes.add(code)
                
                # Clean up the title - remove common suffixes and patterns
                title = re.sub(r'\s*\([^)]*\)\s*$', '', title)  # Remove (3 units)
                title = re.sub(r'\s*\d+\.?\d*\s*CU\s*$', '', title, re.IGNORECASE)  # Remove 3.0 CU
                title = re.sub(r'\s*\d+\.\d+\.\s*.*$', '', title)  # Remove "1.1. Pay Units: 0.0"
                title = re.sub(r'\s+', ' ', title.strip())  # Normalize spaces
                
                # Only add if title is substantial (more than just a few characters)
                if len(title) > 3:
                    subject_data = {
                        'code': code,
                        'title': title,
                        'full': f"{code} - {title}",
                        'year': current_year,
                        'semester': current_semester
                    }
                    subjects.append(subject_data)
                    print(f"📚 Extracted: {code} - {title} (Year {current_year}, {current_semester or 'unknown'} semester)")
        
        print(f"📊 Total subjects extracted: {len(subjects)}")
        return subjects

    def _is_false_positive_match(self, search_term: str, subject_text: str) -> bool:
        """Detect false positive matches to avoid irrelevant results, but be context-aware"""
        
        search_term_lower = search_term.lower()
        subject_text_lower = subject_text.lower()
        
        # Don't filter service programs if the user is specifically searching for them
        if search_term_lower in ['nstp', 'integration', 'service', 'training program', 'national service']:
            return False
        
        # Define false positive patterns - more comprehensive list
        false_positive_patterns = {
            'programming': [
                'national service training program',
                'seniors integration program', 
                'seniors\' integration program',
                'senior integration program',
                'civic welfare training program',
                'nstp program',
                'training program',
                'integration program',
                'asf 4204',  # Specific course code for seniors integration program
                'asf 1102',  # Specific course code for national service training
                'asf 1203'   # Another NSTP course code
            ],
            'program': [
                'national service training program',
                'seniors integration program',
                'seniors\' integration program', 
                'civic welfare training program',
                'integration program',
                'asf 4204',
                'asf 1102',
                'asf 1203'
            ],
            'research': [
                'national service training program',  # Avoid matching "training" as research
                'seniors integration program'
            ],
            'training': [
                'national service training program',  # This is civic training, not academic training
                'civic welfare training'
            ],
            # Don't add entries for service-related terms
        }
        
        # Check term-specific false positives
        if search_term_lower in false_positive_patterns:
            for false_pattern in false_positive_patterns[search_term_lower]:
                if false_pattern in subject_text_lower:
                    return True
        
        # Only apply general false positives for non-service searches
        if not any(term in search_term_lower for term in ['nstp', 'integration', 'service', 'training']):
            # Additional general false positive patterns for any subject
            general_false_positives = [
                'seniors integration program',
                'seniors\' integration program',
                'national service training program',
                'civic welfare training program',
                'asf 4204',
                'asf 1102',
                'asf 1203'
            ]
            
            # Check general false positives that apply to any search term
            for false_pattern in general_false_positives:
                if false_pattern in subject_text_lower:
                    # Return true only if the similarity would be borderline (like 0.362)
                    # This prevents filtering out legitimate matches with high similarity
                    return True
        
        return False
    
    def _find_programs_with_subject_semantic(self, program_subject_map: dict, subject_info: dict, is_negative: bool) -> dict:
        """Find programs that have (or don't have) the specified subject using AI semantic matching"""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        target_subject = subject_info['subject'].lower()
        matching_programs = {}
        
        print(f"🤖 Using AI semantic matching for subject: '{target_subject}'")
        
        # Enhanced cross-term awareness for academic final requirements
        cross_term_mappings = {
            'thesis': ['capstone', 'research project', 'final project', 'dissertation'],
            'capstone': ['thesis', 'research project', 'final project', 'design project'],
            'research project': ['thesis', 'capstone', 'final project', 'research'],
            'final project': ['thesis', 'capstone', 'research project', 'project study'],
            'practicum': ['internship', 'fieldwork', 'clinical', 'teaching practice'],
            'internship': ['practicum', 'fieldwork', 'work experience', 'training'],
            'ojt': ['practicum', 'internship', 'work experience', 'industry practice'],
            # Add math-related terms
            'math': ['calculus', 'statistics', 'algebra', 'geometry', 'trigonometry', 'differential', 'integral', 'linear algebra', 'discrete mathematics', 'mathematics', 'statistical', 'probability'],
            'mathematics': ['calculus', 'statistics', 'algebra', 'geometry', 'trigonometry', 'differential', 'integral', 'linear algebra', 'discrete mathematics', 'math', 'statistical', 'probability'],
            'calculus': ['math', 'mathematics', 'differential', 'integral', 'derivative'],
            'statistics': ['math', 'mathematics', 'statistical', 'probability', 'statistical analysis'],
            'algebra': ['math', 'mathematics', 'linear algebra', 'abstract algebra'],
            'geometry': ['math', 'mathematics', 'trigonometry']
        }
        
        # Get related terms for broader matching
        search_terms = [target_subject]
        if target_subject in cross_term_mappings:
            search_terms.extend(cross_term_mappings[target_subject])
            print(f"🔗 Cross-term matching enabled. Searching for: {search_terms}")
        
        # Vectorize all search terms using both TF-IDF and Word2Vec
        all_target_vectors = []
        for term in search_terms:
            try:
                # Dual preprocessing approach for optimal performance
                # TF-IDF: Use stemmed text (consistent with training corpus)
                term_stemmed = " ".join(preprocess_text(term))
                # Word2Vec: Use raw text (no stemming, as Word2Vec expects full words)
                term_raw = term
                
                # TF-IDF vectorization (using stemmed text)
                tfidf_vector = None
                if hasattr(self, 'tfidf_vectorizer') and self.tfidf_vectorizer is not None:
                    tfidf_vector = self.tfidf_vectorizer.transform([term_stemmed])
                    tfidf_vector = sparse_to_array(tfidf_vector)
                
                # Word2Vec vectorization (using raw tokens)
                w2v_vector = None
                if hasattr(self, 'word2vec_model') and self.word2vec_model is not None:
                    raw_tokens = term_raw.split()
                    w2v_vector = compute_word2vec_vector(raw_tokens, self.word2vec_model)
                
                all_target_vectors.append({
                    'term': term,
                    'tfidf': tfidf_vector,
                    'w2v': w2v_vector,
                    'weight': 1.0 if term == target_subject else 0.7  # Primary term gets full weight
                })
            except Exception as e:
                print(f"⚠️ Error vectorizing term '{term}': {e}")
        
        for program, subjects in program_subject_map.items():
            program_matches = []
            
            for subject in subjects:
                subject_text = f"{subject['code']} {subject['title']}".lower()
                
                # Calculate semantic similarity against all search terms
                max_similarity_score = 0.0
                best_matching_term = target_subject
                
                for target_vector in all_target_vectors:
                    similarity_score = 0.0
                    term = target_vector['term']
                    weight = target_vector['weight']
                    
                    try:
                        # TF-IDF similarity (60% weight) - using stemmed text
                        if target_vector['tfidf'] is not None:
                            subject_stemmed = " ".join(preprocess_text(subject_text))
                            subject_tfidf_vector = self.tfidf_vectorizer.transform([subject_stemmed])
                            subject_tfidf_vector = sparse_to_array(subject_tfidf_vector)
                            tfidf_similarity = cosine_similarity(target_vector['tfidf'], subject_tfidf_vector)[0][0]
                            similarity_score += tfidf_similarity * 0.6
                        
                        # Word2Vec similarity (40% weight) - using raw text
                        if target_vector['w2v'] is not None:
                            subject_raw_tokens = subject_text.split()
                            subject_w2v_vector = compute_word2vec_vector(subject_raw_tokens, self.word2vec_model)
                            if subject_w2v_vector is not None and target_vector['w2v'] is not None:
                                w2v_similarity = cosine_similarity(
                                    target_vector['w2v'].reshape(1, -1), 
                                    subject_w2v_vector.reshape(1, -1)
                                )[0][0]
                                similarity_score += w2v_similarity * 0.4
                        
                        # Enhanced keyword matching with partial matches and false positive filtering
                        if term in subject_text:
                            # Check for false positives (e.g., "programming" matching "program")
                            if self._is_false_positive_match(term, subject_text):
                                # Service programs like "SENIORS INTEGRATION PROGRAM" get almost no boost
                                # This will drop them below the threshold
                                similarity_score = similarity_score * 0.5  # Reduce existing score by half
                            else:
                                similarity_score += 0.4  # Boost for exact matches
                        else:
                            # Check for partial word matches (e.g., "math" matches "mathematics")
                            term_words = term.split()
                            for word in term_words:
                                if len(word) > 3 and word in subject_text:
                                    # Check for false positives in partial matches too
                                    if not self._is_false_positive_match(word, subject_text):
                                        similarity_score += 0.2  # Smaller boost for partial matches
                        
                        # Apply cross-term weight and check if this is the best match
                        weighted_score = similarity_score * weight
                        if weighted_score > max_similarity_score:
                            max_similarity_score = weighted_score
                            best_matching_term = term
                        
                    except Exception as e:
                        print(f"⚠️ Error calculating similarity for {subject_text} vs {term}: {e}")
                        # Fallback to simple string matching
                        if term in subject_text:
                            fallback_score = 0.8 * weight
                            if fallback_score > max_similarity_score:
                                max_similarity_score = fallback_score
                                best_matching_term = term
                
                # Only filter service programs if the search is NOT specifically for them
                # This way, if someone asks about NSTP, they'll get results
                if target_subject not in ['nstp', 'integration', 'service', 'training program', 'national service']:
                    # Check if this is a service program that should be excluded from non-service searches
                    service_program_indicators = [
                        'asf 4204', 'asf 1102', 'asf 1203',
                        'seniors integration program', "seniors' integration program",
                        'national service training program', 'civic welfare training'
                    ]
                    
                    # Skip this match only if it's a service program AND we're not looking for service programs
                    if any(indicator in subject_text.lower() for indicator in service_program_indicators):
                        continue
                
                # Set threshold for matches (0.35 to filter out false positives)
                if max_similarity_score > 0.35:
                    program_matches.append({
                        'subject': subject,
                        'similarity': max_similarity_score,
                        'matched_term': best_matching_term
                    })
                    match_indicator = "🔗" if best_matching_term != target_subject else "✅"
                    print(f"{match_indicator} Match: {program} - {subject['full']} (similarity: {max_similarity_score:.3f}, term: {best_matching_term})")
        
            # Apply positive/negative logic
            if is_negative:
                if not program_matches:  # Programs that DON'T have the subject
                    matching_programs[program] = []
            else:
                if program_matches:  # Programs that DO have the subject
                    # Sort by similarity and keep the matches
                    program_matches.sort(key=lambda x: x['similarity'], reverse=True)
                    matching_programs[program] = [match['subject'] for match in program_matches]
        
        print(f"🎯 Found {len(matching_programs)} matching programs")
        return matching_programs

    def _get_program_school_from_config(self, program_acronym: str) -> str:
        """Get school name for a program using the JSON config"""
        try:
            config = self._load_normalization_config()
            program_abbrevs = config.get('program_abbreviations', {})
            
            # Search through all school sections
            for school_section, programs in program_abbrevs.items():
                if school_section.startswith('_'):  # Skip metadata
                    continue
                    
                for abbrev_key, abbrev_data in programs.items():
                    # Check if this matches our program
                    full_name = abbrev_data.get('full_name', '').upper()
                    if full_name == program_acronym.upper():
                        school = abbrev_data.get('school', '')
                        print(f"🔍 Found school for {program_acronym}: {school}")
                        return school
            
            print(f"⚠️ No school found for program: {program_acronym}")
            return None
        except Exception as e:
            print(f"⚠️ Error getting school from config: {e}")
            return None
            
    def _get_program_cluster_from_config(self, program_acronym: str) -> str:
        """Get cluster name for a program using the JSON config"""
        try:
            config = self._load_normalization_config()
            program_abbrevs = config.get('program_abbreviations', {})
            
            # Search through all school sections
            for school_section, programs in program_abbrevs.items():
                if school_section.startswith('_'):  # Skip metadata
                    continue
                    
                for abbrev_key, abbrev_data in programs.items():
                    # Check if this matches our program
                    full_name = abbrev_data.get('full_name', '').upper()
                    if full_name == program_acronym.upper():
                        cluster = abbrev_data.get('cluster', '')
                        print(f"🔍 Found cluster for {program_acronym}: {cluster}")
                        return cluster
            
            print(f"⚠️ No cluster found for program: {program_acronym}")
            return None
        except Exception as e:
            print(f"⚠️ Error getting cluster from config: {e}")
            return None
    
    def _get_program_full_name_from_config(self, program_acronym: str) -> str:
        """Get full program name/description for a program using the JSON config"""
        try:
            config = self._load_normalization_config()
            program_abbrevs = config.get('program_abbreviations', {})
            
            # Search through all school sections
            for school_section, programs in program_abbrevs.items():
                if school_section.startswith('_'):  # Skip metadata
                    continue
                    
                for abbrev_key, abbrev_data in programs.items():
                    # Check if this matches our program
                    full_name = abbrev_data.get('full_name', '').upper()
                    if full_name == program_acronym.upper():
                        description = abbrev_data.get('description', '')
                        print(f"🔍 Found description for {program_acronym}: {description}")
                        return description
            
            print(f"⚠️ No description found for program: {program_acronym}")
            return program_acronym  # Fallback to acronym if no description found
        except Exception as e:
            print(f"⚠️ Error getting description from config: {e}")
            return program_acronym

    def _apply_cluster_filter_to_curriculum_docs(self, curriculum_docs: List[Dict], cluster_filter: str) -> List[Dict]:
        """Filter curriculum documents by cluster using JSON config mapping"""
        if not cluster_filter:
            return curriculum_docs
        
        filtered_docs = []
        
        for doc in curriculum_docs:
            # Extract program acronym from curriculum content
            program_acronym = self._extract_program_acronym_from_curriculum(doc['content'])
            
            if program_acronym:
                # Get cluster from JSON config
                doc_cluster = self._get_program_cluster_from_config(program_acronym)
                
                # Check if it matches the filter
                if doc_cluster and cluster_filter.lower() in doc_cluster.lower():
                    filtered_docs.append(doc)
                    print(f"✅ Including {program_acronym} (cluster: {doc_cluster})")
                else:
                    print(f"🚫 Excluding {program_acronym} (cluster: {doc_cluster}, filter: {cluster_filter})")
            else:
                print(f"⚠️ Could not extract program acronym from document")
        
        return filtered_docs

    def _clean_ai_response(self, response: str) -> str:
        """Clean AI response from malformed tags and duplicate content"""
        if not response:
            return response
        
        # Remove any AI assistant tags that might have leaked through
        response = response.replace('<|assistant|>', '')
        response = response.replace('</|assistant|>', '')
        response = response.replace('<|system|>', '')
        response = response.replace('</|system|>', '')
        response = response.replace('<|user|>', '')
        response = response.replace('</|user|>', '')
        
        # Remove duplicate "assistant" text that appears at the end
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            # Skip lines that are just "assistant" or similar AI artifacts
            if stripped_line.lower() in ['assistant', '<|assistant|>', '</|assistant|>']:
                continue
            cleaned_lines.append(line)
        
        # Join back and clean up extra whitespace
        cleaned_response = '\n'.join(cleaned_lines).strip()
        
        # Remove any duplicate content by splitting on common patterns
        # If the response appears to be duplicated, take only the first occurrence
        if 'Here are the programs that' in cleaned_response:
            parts = cleaned_response.split('Here are the programs that')
            if len(parts) > 2:  # More than one occurrence
                # Keep the first complete response
                first_part = 'Here are the programs that' + parts[1]
                # Find where the first response ends (usually at "For more detailed information")
                if 'For more detailed information' in first_part:
                    end_index = first_part.find('For more detailed information')
                    cleaned_response = first_part[:end_index + len('For more detailed information, please contact the admissions office.')]
        
        return cleaned_response
    
    def _format_subject_mapping_response_text(self, response_text: str) -> str:
        """Post-process subject mapping response to ensure proper formatting"""
        import re
        
        # CRITICAL: Fix programs that are on the same line (most common issue)
        # Look for patterns like "• **Program 1**: courses • **Program 2**: courses"
        if '• **' in response_text:
            # Replace bullet points that appear mid-line with newline + bullet
            response_text = re.sub(r'([^•\n])\s*•\s*\*\*', r'\1\n• **', response_text)
            
            # Also handle cases where there's no space before the bullet
            response_text = re.sub(r'([^•\n])•\s*\*\*', r'\1\n• **', response_text)
        
        # If the response doesn't have proper line breaks, fix it
        if '•' in response_text and response_text.count('\n•') < response_text.count('•') - 1:
            # Split on bullet points and rejoin with proper line breaks
            parts = response_text.split('•')
            if len(parts) > 1:
                intro = parts[0].strip()
                program_lines = []
                
                for part in parts[1:]:
                    if part.strip():
                        # Clean up the part and ensure it starts properly
                        cleaned_part = part.strip()
                        if not cleaned_part.startswith('**'):
                            # Find where the program name starts
                            if '**' in cleaned_part:
                                cleaned_part = cleaned_part[cleaned_part.find('**'):]
                        program_lines.append(f"• {cleaned_part}")
                
                # Find conclusion (usually starts with "For more detailed")
                conclusion = ""
                if program_lines:
                    last_line = program_lines[-1]
                    if "For more detailed" in last_line:
                        # Extract conclusion from last line
                        conclusion_match = re.search(r'(For more detailed.*)', last_line)
                        if conclusion_match:
                            conclusion = conclusion_match.group(1)
                            # Remove conclusion from last program line
                            program_lines[-1] = re.sub(r'\s*For more detailed.*', '', last_line)
                
                # Reconstruct with proper formatting
                formatted_response = intro
                if program_lines:
                    formatted_response += "\n\n" + "\n".join(program_lines)
                if conclusion:
                    formatted_response += "\n\n" + conclusion
                
                return formatted_response
        
        # Additional cleanup: ensure each bullet point is on its own line
        lines = response_text.split('\n')
        new_lines = []
        
        for line in lines:
            # Check if line has multiple bullet points
            if line.count('• **') > 1:
                # Split this line into multiple program lines
                # Use a more sophisticated split that preserves the program content
                bullet_positions = []
                start = 0
                while True:
                    pos = line.find('• **', start)
                    if pos == -1:
                        break
                    bullet_positions.append(pos)
                    start = pos + 1
                
                # Extract each program segment
                for i, pos in enumerate(bullet_positions):
                    if i == len(bullet_positions) - 1:
                        # Last segment - take everything from this position to end
                        segment = line[pos:].strip()
                    else:
                        # Take from this position to just before next bullet
                        next_pos = bullet_positions[i + 1]
                        segment = line[pos:next_pos].strip()
                    
                    if segment:
                        new_lines.append(segment)
            else:
                new_lines.append(line)
        
        response_text = '\n'.join(new_lines)
        
        # Final cleanup: remove any empty lines between programs (but keep one blank line after intro)
        lines = response_text.split('\n')
        final_lines = []
        intro_done = False
        
        for i, line in enumerate(lines):
            if not intro_done and line.strip() and not line.strip().startswith('•'):
                # This is intro text
                final_lines.append(line)
                if i + 1 < len(lines) and lines[i + 1].strip().startswith('•'):
                    # Next line is a bullet point, add blank line
                    final_lines.append('')
                    intro_done = True
            elif line.strip().startswith('•'):
                # This is a program line
                final_lines.append(line)
                intro_done = True
            elif line.strip() and 'For more detailed' in line:
                # This is the conclusion
                if final_lines and final_lines[-1].strip():
                    final_lines.append('')  # Add blank line before conclusion
                final_lines.append(line)
            elif line.strip():
                # Other non-empty lines
                final_lines.append(line)
        
        return '\n'.join(final_lines)
    
    def _format_subject_mapping_response(self, matching_programs: dict, subject_info: dict, is_negative: bool, cluster_filter: str = None) -> List[Dict]:
        """Format the response for subject mapping queries based on query type"""
        
        subject = subject_info['subject']
        query_type = subject_info.get('query_type', 'general')
        program = subject_info.get('program')
        
        # Handle empty results first
        if not matching_programs:
            if query_type in ['program_specific', 'count']:
                if program:
                    response_text = f"I couldn't find any {subject} courses in the {program} program. This subject may not be offered in this program."
                else:
                    response_text = f"I couldn't find the specified program. Please check the program name and try again."
            elif is_negative:
                response_text = f"I couldn't find any programs that don't offer {subject}. All programs may include this subject."
            else:
                response_text = f"I couldn't find any programs that offer {subject}. Please try a different subject or check the spelling."
            
            return [{
                'id': 'subject_mapping_result',
                'content': response_text,
                'relevance': 1.0,
                'retrieval_strategy': 'subject_mapping',
                'matching_programs': 0,
                'subject_searched': subject,
                'query_type': query_type
            }]
        
        # Handle different query types
        if query_type == 'count':
            # Count query: How many X subjects in program Y?
            program_name = list(matching_programs.keys())[0]  # Should only be one program
            subjects = matching_programs[program_name]
            count = len(subjects)
            
            response_text = f"**{program_name}** offers {count} {subject} courses:\n\n"
            for subj in subjects:
                response_text += f"• {subj['full']}\n"
                
        elif query_type in ['program_specific', 'negative_program_specific']:
            # Program-specific query: What X subjects are in program Y?
            program_name = list(matching_programs.keys())[0]  # Should only be one program
            subjects = matching_programs[program_name]
            
            if query_type == 'negative_program_specific':
                if not subjects:
                    response_text = f"**{program_name}** does not offer any {subject} courses."
                else:
                    response_text = f"**{program_name}** does not have these specific {subject} courses, but it offers other related courses:\n\n"
                    for subj in subjects:
                        response_text += f"• {subj['full']}\n"
            else:
                if subjects:
                    response_text = f"Here are the {subject} courses offered in **{program_name}**:\n\n"
                    for subj in subjects:
                        response_text += f"• {subj['full']}\n"
                else:
                    response_text = f"**{program_name}** does not offer any {subject} courses."
                    
        elif is_negative:
            # Negative query: What programs don't have X?
            response_text = f"These programs do not offer {subject}:\n\n"
            for program in matching_programs.keys():
                response_text += f"• **{program}**\n"
                
        else:
            # General positive query: What programs have X?
            # This is handled by the AI prompt for better school/cluster organization
            cluster_text = f" under {cluster_filter} cluster" if cluster_filter else ""
            response_text = f"Here are the programs{cluster_text} that offer {subject}:\n\n"
            
            # Let the AI organize by school/cluster using the prompt
            # Just provide the raw data here
            for program, subjects in matching_programs.items():
                response_text += f"• **{program}**: "
                subject_texts = [s['full'] for s in subjects]
                response_text += ", ".join(subject_texts) + "\n"
        
        return [{
            'id': 'subject_mapping_result',
            'content': response_text,
            'relevance': 1.0,
            'retrieval_strategy': 'subject_mapping',
            'matching_programs': len(matching_programs),
            'subject_searched': subject,
            'query_type': query_type,
            'is_negative': is_negative,
            'cluster_filter': cluster_filter
        }]

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

    def retrieve_documents_by_topic_hybrid(self, query: str, topic_id: str, top_k: int = 2, cluster_filter: str = None) -> List[Dict]:
        """
        HYBRID TOPIC + SEMANTIC RETRIEVAL:
        1. Filter documents by topic keywords (same as before)
        2. Generate TF-IDF + Word2Vec vectors for query and filtered documents
        3. Calculate semantic similarity using cosine similarity
        4. Combine topic relevance (60%) + semantic similarity (40%)
        """
        import re
        
        print(f"🔬 Hybrid topic + semantic retrieval for: '{query}' (topic: {topic_id})")
        
        # Normalize program acronyms and school abbreviations in query for better matching
        normalized_query = self._normalize_program_acronyms(query)
        if normalized_query != query:
            print(f"📝 Program normalized query: '{query}' → '{normalized_query}'")
        
        # Also normalize school abbreviations
        school_normalized_query = self._normalize_school_abbreviations(normalized_query)
        if school_normalized_query != normalized_query:
            print(f"📝 School normalized query: '{normalized_query}' → '{school_normalized_query}'")
            normalized_query = school_normalized_query
        
        try:
            collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)
            
            # STAGE 1: Topic-based filtering with document type filtering
            # Use JSON-based keywords for programs, regular topic keywords for others
            topic_keywords = self._extract_keywords_from_json(topic_id)
            if not topic_keywords:
                print(f"⚠️ No keywords found for topic: {topic_id}, falling back to simple retrieval")
                return self.retrieve_documents_by_topic_keywords_simple(query, topic_id, top_k)
            
            print(f"📝 Keywords for filtering: {len(topic_keywords)} keywords ({'JSON-based' if topic_id == 'programs_courses' else 'topic-based'})")
            
            # Get topic-specific document types for filtering
            # Map topic_id to strategy name
            try:
                from .topics import get_retrieval_strategy_config
            except ImportError:
                from topics import get_retrieval_strategy_config
            strategy_mapping = {
                'programs_courses': 'programs_specialized',
                'admissions_enrollment': 'admissions_specialized', 
                'fees': 'fees_specialized'
            }
            strategy_name = strategy_mapping.get(topic_id, f'{topic_id}_specialized')
            strategy_config = get_retrieval_strategy_config(strategy_name)
            document_types = strategy_config.get('document_types', [])
            print(f"📋 Document types for {topic_id}: {document_types}")
            
            # Build where clause with document type filtering
            if document_types:
                where_clause = {
                    "$and": [
                        {"source": "pdf_scrape"},
                        {"$or": [{"document_type": doc_type} for doc_type in document_types]}
                    ]
                }
                print(f"🔍 Filtering by document types: {document_types}")
            else:
                where_clause = {"source": "pdf_scrape"}
                print(f"⚠️ No document type filtering for {topic_id}")
            
            # Get documents with proper filtering
            all_docs = collection.get(
                where=where_clause,
                include=["documents", "metadatas"]
            )
            
            all_ids = all_docs.get('ids', [])
            all_contents = all_docs.get('documents', [])
            all_metadatas = all_docs.get('metadatas', [])
            
            # Filter fees documents: only allow regulation/policy fees documents for admissions
            if topic_id == 'admissions_enrollment':
                filtered_ids = []
                filtered_contents = []
                filtered_metadatas = []
                
                for doc_id, content, metadata in zip(all_ids, all_contents, all_metadatas):
                    doc_type = metadata.get('document_type', '').lower()
                    
                    # If it's a fees document, check if it's a regulation/policy
                    if doc_type in ['fees', 'financial']:
                        doc_keywords = metadata.get('keywords', '').lower()
                        filename = metadata.get('filename', '').lower()
                        
                        # Only include if it's a regulation/policy document
                        is_regulation = (
                            'regulation' in doc_keywords or 
                            'regulation' in filename or
                            'policy' in doc_keywords or
                            'policy' in filename or
                            filename.endswith('.pdf')  # PDFs are likely policies, CSVs are program fees
                        )
                        
                        if is_regulation:
                            filtered_ids.append(doc_id)
                            filtered_contents.append(content)
                            filtered_metadatas.append(metadata)
                            print(f"✅ Including fees regulation document: {metadata.get('filename', 'N/A')}")
                        else:
                            print(f"🚫 Excluding non-regulation fees document: {metadata.get('filename', 'N/A')}")
                    else:
                        # Include all non-fees documents
                        filtered_ids.append(doc_id)
                        filtered_contents.append(content)
                        filtered_metadatas.append(metadata)
                
                # Update the lists with filtered results
                all_ids = filtered_ids
                all_contents = filtered_contents
                all_metadatas = filtered_metadatas
                print(f"📚 After fees regulation filtering: {len(all_ids)} documents (excluded {len(all_docs.get('ids', [])) - len(all_ids)} non-regulation fees docs)")
            
            print(f"📚 Applying topic keyword filtering to {len(all_ids)} document type-filtered documents...")
            
            # Filter documents by topic keywords
            topic_filtered_docs = []
            for i, (doc_id, content, metadata) in enumerate(zip(all_ids, all_contents, all_metadatas)):
                doc_keywords = metadata.get('keywords', '').lower()
                filename = metadata.get('filename', '').lower()
                content_lower = content.lower()
                
                # Calculate topic relevance score
                topic_score = 0.0
                matched_keywords = []
                
                for topic_keyword in topic_keywords:
                    topic_keyword_lower = topic_keyword.lower()
                    pattern = r'\b' + re.escape(topic_keyword_lower) + r'\b'
                    
                    # Check in keywords, filename, and content (with different weights)
                    if re.search(pattern, doc_keywords):
                        topic_score += 3  # High weight for metadata keywords
                        matched_keywords.append(f"{topic_keyword}(meta)")
                    elif re.search(pattern, filename):
                        topic_score += 2  # Medium weight for filename
                        matched_keywords.append(f"{topic_keyword}(file)")
                    elif re.search(pattern, content_lower):
                        topic_score += 1  # Low weight for content
                        matched_keywords.append(f"{topic_keyword}(content)")
                
                # Only include documents with topic relevance
                if topic_score > 0:
                    # Normalize topic score
                    max_possible_score = len(topic_keywords) * 3
                    normalized_topic_score = topic_score / max_possible_score if max_possible_score > 0 else 0.0
                    
                    topic_filtered_docs.append({
                        'id': doc_id,
                        'content': content,
                        'metadata': metadata,
                        'topic_score': normalized_topic_score,
                        'matched_keywords': matched_keywords
                    })
            
            print(f"🎯 Found {len(topic_filtered_docs)} documents matching topic keywords")
            
            # STAGE 1.2: Apply keyword fallback for all topics with topic-specific exclusions
            print(f"🔄 Applying keyword fallback for {topic_id} (found {len(topic_filtered_docs)} topic-filtered documents)")
            
            # Expand search to ALL documents
            all_docs_expanded = collection.get(
                where={"source": "pdf_scrape"},
                include=["documents", "metadatas"]
            )
            
            expanded_ids = all_docs_expanded.get('ids', [])
            expanded_contents = all_docs_expanded.get('documents', [])
            expanded_metadatas = all_docs_expanded.get('metadatas', [])
            
            print(f"📚 Expanded search to {len(expanded_ids)} total documents")
            
            # Filter by topic keywords in metadata/filename with topic-specific exclusions
            fallback_docs = []
            for i, (doc_id, content, metadata) in enumerate(zip(expanded_ids, expanded_contents, expanded_metadatas)):
                # Skip if already included
                if any(existing_doc['id'] == doc_id for existing_doc in topic_filtered_docs):
                    continue
                
                doc_keywords = metadata.get('keywords', '').lower()
                filename = metadata.get('filename', '').lower()
                
                # Check if document has topic-related keywords
                has_topic_keywords = False
                matched_keywords = []
                
                for keyword in topic_keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in doc_keywords or keyword_lower in filename:
                        has_topic_keywords = True
                        matched_keywords.append(f"{keyword}(fallback)")
                        break
                    
                # Include if it has topic keywords AND passes topic-specific exclusion rules
                if has_topic_keywords:
                    doc_type = metadata.get('document_type', '').lower()
                    
                    # Topic-specific exclusion rules
                    should_exclude = False
                    exclusion_reason = None
                    
                    if topic_id == 'programs_courses':
                        # Programs: Exclude fees, admissions, enrollment, etc.
                        # Only allow: academic, curriculum (or empty for legacy docs)
                        excluded_types = ['fees', 'financial', 'admission', 'enrollment', 'scholarship', 'policy', 'contact', 'other']
                        if doc_type and doc_type in excluded_types:
                            should_exclude = True
                            exclusion_reason = f"{doc_type} document type not for programs"
                    
                    elif topic_id == 'admissions_enrollment':
                        # Admissions: Exclude programs, but ALLOW fees regulations/policies
                        # Exclude: academic, curriculum
                        # Allow: admission, enrollment, scholarship, policy, contact, fees (if regulation/policy)
                        excluded_types = ['academic', 'curriculum']
                        
                        if doc_type in excluded_types:
                            should_exclude = True
                            exclusion_reason = f"{doc_type} document type not for admissions"
                        elif doc_type in ['fees', 'financial']:
                            # For fees documents in admissions: only allow if they're about regulations/policies
                            # Check if it's a regulation/policy document (not program fees CSV)
                            is_regulation = (
                                'regulation' in doc_keywords or 
                                'regulation' in filename or
                                'policy' in doc_keywords or
                                'policy' in filename or
                                filename.endswith('.pdf')  # PDFs are likely policies, CSVs are program fees
                            )
                            if not is_regulation:
                                # This is likely a program fees CSV (e.g., SAS-FEES-1.csv)
                                should_exclude = True
                                exclusion_reason = "program fees document (only fees regulations allowed in admissions)"
                    
                    elif topic_id == 'fees':
                        # Fees: Exclude programs, admissions, but allow ALL fees documents
                        # Exclude: academic, curriculum, admission, enrollment, scholarship, contact
                        excluded_types = ['academic', 'curriculum', 'admission', 'enrollment', 'scholarship', 'contact', 'other']
                        if doc_type in excluded_types:
                            should_exclude = True
                            exclusion_reason = f"{doc_type} document type not for fees"
                        # Allow: fees, financial (both regulations AND program fees)
                    
                    if should_exclude:
                        print(f"🚫 Excluding from {topic_id}: {metadata.get('filename', 'N/A')} ({exclusion_reason})")
                        continue
                    
                        # Calculate basic topic score for fallback docs
                        topic_score = len(matched_keywords) / len(topic_keywords) if topic_keywords else 0.0
                        
                        fallback_docs.append({
                            'id': doc_id,
                            'content': content,
                            'metadata': metadata,
                            'topic_score': topic_score,
                            'matched_keywords': matched_keywords
                        })
                        print(f"✅ Including fallback document: {metadata.get('filename', 'N/A')}")
                
                # Add fallback docs to topic_filtered_docs
                topic_filtered_docs.extend(fallback_docs)
                print(f"✅ After keyword fallback: {len(topic_filtered_docs)} documents total")
            
            # STAGE 1.5: Apply cluster filtering if specified
            if cluster_filter:
                print(f"🎯 Applying cluster filtering for: '{cluster_filter}'")
                cluster_filtered_docs = []
                
                # Check if this is a school-level query (no cluster filtering needed)
                school_abbreviations = self._get_school_abbreviations()
                is_school_query = False
                for abbrev, school_name in school_abbreviations.items():
                    if abbrev.lower() in normalized_query.lower() or school_name.lower() in normalized_query.lower():
                        is_school_query = True
                        print(f"🏫 Detected school-level query: {school_name}")
                        break
                
                if is_school_query:
                    # For school-level queries, don't apply cluster filtering
                    print(f"🎯 School-level query detected, skipping cluster filtering")
                    print(f"🎯 Using all topic-filtered documents: {len(topic_filtered_docs)} documents")
                else:
                    # Get programs in this cluster from JSON config
                    cluster_programs = self._get_programs_in_cluster(cluster_filter)
                    print(f"📋 Programs in {cluster_filter}: {cluster_programs}")
                    
                    for doc in topic_filtered_docs:
                        metadata = doc.get('metadata', {})
                        filename = metadata.get('filename', '').lower()
                        
                        # Check if document filename matches any program in this cluster
                        matches_cluster = False
                        for program in cluster_programs:
                            if program.lower() in filename:
                                matches_cluster = True
                                break
                        
                        if matches_cluster:
                            cluster_filtered_docs.append(doc)
                            print(f"  ✅ {metadata.get('filename', 'Unknown')} - Matches cluster programs")
                        else:
                            print(f"  ❌ {metadata.get('filename', 'Unknown')} - No cluster program match (filtered out)")
                    
                    topic_filtered_docs = cluster_filtered_docs
                    print(f"🎯 After cluster filtering: {len(topic_filtered_docs)} documents")
            
            if not topic_filtered_docs:
                print("⚠️ No documents found after filtering")
                return []
            
            # STAGE 2: Apply specialized logic based on topic
            print(f"🎯 Applying specialized logic for topic: {topic_id}")
            
            specialized_scored_docs = []
            
            if topic_id == 'programs_courses':
                # Apply programs-specific logic with JSON-based enhancements
                
                # Detect cluster/school/general query type using JSON first
                query_context = self._detect_cluster_or_school_in_query(normalized_query)
                print(f"🎯 Query context: {query_context}")
                
                # Check if this is an overview query (don't extract specific program)
                overview_keywords = ['available', 'list', 'show', 'all programs', 'what are the programs', 'what programs']
                normalized_lower = normalized_query.lower()
                has_overview_keyword = any(word in normalized_lower for word in overview_keywords)
                is_general_query = query_context['type'] == 'general'
                
                # Additional check: if query contains overview keywords AND doesn't mention a specific program, it's overview
                # Extract a test program_info to see if it would match anything
                # But only do this as a last check - overview keywords should be sufficient
                is_overview_query = is_general_query and has_overview_keyword
                
                # If it looks like overview, verify it's not actually asking about a specific program
                # by checking if it would extract a program with high confidence
                if is_overview_query:
                    # Quick check: if overview keywords are present, assume it's overview
                    # unless there's a very explicit program mention (like "BS CS" or full program name)
                    test_extraction = self._extract_program_info(normalized_query)
                    if test_extraction.get('program_name'):
                        # If a program was extracted, check if it's a false positive
                        # Overview queries shouldn't extract programs unless explicitly mentioned
                        # Check if the query has explicit program mentions (abbreviations, full names)
                        explicit_program_patterns = [
                            r'\bbs\s+\w+',  # "BS CS", "BS IT"
                            r'\bab\s+\w+',  # "AB IS", "AB ENG"
                            r'bachelor\s+of',  # "Bachelor of Science"
                            r'\b\w+\s+program\b',  # "computer science program"
                        ]
                        has_explicit_program = any(
                            re.search(pattern, normalized_lower, re.IGNORECASE) 
                            for pattern in explicit_program_patterns
                        )
                        
                        if not has_explicit_program:
                            # This is likely a false positive match - treat as overview
                            print(f"⚠️ Program extraction found '{test_extraction.get('program_name')}' but no explicit program mention - treating as overview")
                            is_overview_query = True
                        else:
                            # Has explicit program mention - not an overview query
                            print(f"ℹ️ Has explicit program mention - not treating as overview")
                            is_overview_query = False
                
                print(f"🔍 Overview detection:")
                print(f"  - normalized_query: '{normalized_query}'")
                print(f"  - query_context['type']: '{query_context['type']}'")
                print(f"  - has_overview_keyword: {has_overview_keyword}")
                print(f"  - is_overview_query (final): {is_overview_query}")
                
                if is_overview_query:
                    # Don't extract a specific program for overview queries
                    program_info = {
                        'program_name': None,
                        'degree_level': None,
                        'year_level': None,
                        'course_code': None,
                        'context_source': None
                    }
                    print(f"📋 Overview query detected - skipping program extraction")
                else:
                    # Extract program info for specific program queries
                    program_info = self._extract_program_info(normalized_query)
                
                print(f"📚 Extracted program info: {program_info}")
                
                # Classify query intent for programs topic
                query_intent = None
                if topic_id == 'programs_courses':
                    query_intent = self._classify_programs_query_intent(normalized_query)
                    print(f"🎯 Classified query intent: {query_intent}")
                
                # Get relevant programs from JSON based on query context
                relevant_programs = []
                if query_context['type'] == 'cluster':
                    relevant_programs = self._get_programs_by_cluster_from_json(query_context['value'])
                elif query_context['type'] == 'school':
                    relevant_programs = self._get_programs_by_school_from_json(query_context['value'])
                elif query_context['type'] == 'general':
                    all_programs = self._get_all_programs_from_json()
                    relevant_programs = list(all_programs.values())
                
                print(f"📋 Relevant programs for scoring: {len(relevant_programs)}")
                
                # Get strategy configuration for programs
                strategy_config = get_retrieval_strategy_config('programs_specialized')
                priorities = strategy_config.get('metadata_priorities', {})
                
                query_lower = normalized_query.lower()
                
                for doc_data in topic_filtered_docs:
                    metadata = doc_data['metadata']
                    content = doc_data['content']
                    
                    filename = metadata.get('filename', '').lower()
                    doc_keywords = metadata.get('keywords', '').lower()
                    content_lower = content.lower()
                    
                    # Calculate specialized scores using existing methods
                    filename_score = self._calculate_programs_filename_score(filename, program_info, query_intent)
                    keyword_score = self._calculate_keyword_score(doc_keywords, query_lower)
                    content_score = self._calculate_programs_content_score(content_lower, query_lower, program_info, query_intent)
                    
                    # JSON-based program matching boost
                    json_program_boost = 0.0
                    if relevant_programs:
                        for prog in relevant_programs:
                            prog_name = prog.get('full_name', '') if isinstance(prog, dict) else str(prog)
                            prog_abbrev = prog.get('abbrev', '') if isinstance(prog, dict) else ''
                            
                            # Check if document matches this program
                            if (prog_name.lower() in filename or 
                                prog_abbrev.lower() in filename or
                                prog_name.lower() in doc_keywords or
                                prog_abbrev.lower() in doc_keywords):
                                json_program_boost = 0.3  # Boost for JSON program match
                                print(f"🎯 JSON program boost for {prog_name}: {metadata.get('filename', 'N/A')}")
                                break
                    
                    # Apply strategy priorities with JSON boost
                    specialized_score = (
                        filename_score * priorities.get('filename', 0.5) +
                        keyword_score * priorities.get('keywords', 0.3) +
                        content_score * priorities.get('content', 0.2) +
                        json_program_boost
                    )
                    
                    specialized_scored_docs.append({
                        'doc_data': doc_data,
                        'specialized_score': specialized_score,
                        'program_info': program_info,
                        'query_context': query_context,
                        'json_program_boost': json_program_boost
                    })
                    
            elif topic_id == 'fees':
                # Apply fees-specific logic with fee information extraction
                fee_info = self._extract_fee_info(normalized_query)
                print(f"💰 Extracted fee info for fees: {fee_info}")
                
                # Get strategy configuration for fees
                strategy_config = get_retrieval_strategy_config('fees_specialized')
                priorities = strategy_config.get('metadata_priorities', {})
                
                query_lower = normalized_query.lower()
                
                for doc_data in topic_filtered_docs:
                    metadata = doc_data['metadata']
                    content = doc_data['content']
                    
                    filename = metadata.get('filename', '').lower()
                    doc_keywords = metadata.get('keywords', '').lower()
                    content_lower = content.lower()
                    
                    # Calculate specialized scores using existing methods
                    filename_score = self._calculate_fees_filename_score(filename, fee_info)
                    keyword_score = self._calculate_keyword_score(doc_keywords, query_lower)
                    content_score = self._calculate_fees_content_score(content_lower, query_lower, fee_info)
                    
                    # Apply strategy priorities
                    specialized_score = (
                        filename_score * priorities.get('filename', 0.3) +
                        keyword_score * priorities.get('keywords', 0.4) +
                        content_score * priorities.get('content', 0.3)
                    )
                    
                    specialized_scored_docs.append({
                        'doc_data': doc_data,
                        'specialized_score': specialized_score,
                        'program_info': fee_info  # Store fee_info as program_info for consistency
                    })
                    
            else:
                # For other non-programs topics, use topic score as specialized score
                for doc_data in topic_filtered_docs:
                    specialized_scored_docs.append({
                        'doc_data': doc_data,
                        'specialized_score': doc_data['topic_score'],
                        'program_info': None
                    })
            
            print(f"🎯 Applied specialized logic to {len(specialized_scored_docs)} documents")
            
            # STAGE 3: Apply TF-IDF + Word2Vec semantic scoring on specialized results
            print(f"🧠 Applying TF-IDF + Word2Vec semantic scoring to specialized results...")
            
            # Generate query vector using normalized query
            query_vector = self._vectorize_query(normalized_query)
            print(f"📊 Generated query vector: TF-IDF + Word2Vec (dim: {query_vector.shape[0]})")
            
            # Score each specialized document with semantic similarity
            hybrid_scored_results = []
            
            for spec_doc in specialized_scored_docs:
                doc_data = spec_doc['doc_data']
                specialized_score = spec_doc['specialized_score']
                
                # Generate document vector
                doc_vector = self._vectorize_query(doc_data['content'])
                
                # Calculate semantic similarity
                try:
                    from sklearn.metrics.pairwise import cosine_similarity
                    semantic_similarity = cosine_similarity(
                        query_vector.reshape(1, -1),
                        doc_vector.reshape(1, -1)
                    )[0][0]
                except Exception as e:
                    print(f"⚠️ Semantic similarity calculation failed: {e}")
                    semantic_similarity = 0.0
                
                # QUERY-DOCUMENT SPECIFICITY BOOST
                specificity_boost = self._calculate_query_document_specificity(normalized_query, doc_data['metadata'])
                
                # TRUE HYBRID SCORE: 55% semantic similarity + 25% specialized logic + 20% topic + specificity boost
                # MAKE SEMANTIC RETRIEVAL BIG - semantic similarity is now the primary ranking mechanism
                
                # YEAR-CONTEXT AWARENESS: Reduce semantic weight for curriculum follow-up queries
                # to prevent "2nd year" from matching "secondary education"
                if self._is_curriculum_followup_query(query):
                    # Reduce semantic weight and boost specialized logic for curriculum follow-ups
                    semantic_weight = 0.35  # Reduced from 0.55
                    specialized_weight = 0.45  # Increased from 0.25
                    topic_weight = 0.20
                    print(f"🎯 Curriculum follow-up detected - reducing semantic weight to prevent 2nd/secondary confusion")
                else:
                    # Normal weights for other queries
                    semantic_weight = 0.55
                    specialized_weight = 0.25
                    topic_weight = 0.20
                
                semantic_component = semantic_similarity * semantic_weight
                specialized_component = specialized_score * specialized_weight
                topic_component = doc_data['topic_score'] * topic_weight
                final_score = semantic_component + specialized_component + topic_component + specificity_boost
                
                hybrid_scored_results.append({
                    'id': doc_data['id'],
                    'content': doc_data['content'],
                    'relevance': final_score,
                    'folder': doc_data['metadata'].get('folder_name', 'Unknown'),
                    'document_type': doc_data['metadata'].get('document_type', 'other'),
                    'target_program': doc_data['metadata'].get('target_program', 'all'),
                    'filename': doc_data['metadata'].get('filename', ''),
                    'retrieval_strategy': 'hybrid-topic-semantic',
                    'current_topic': topic_id,
                    '_debug': {
                        'specialized_score': specialized_score,
                        'semantic_similarity': semantic_similarity,
                        'topic_score': doc_data['topic_score'],
                        'specialized_component': specialized_component,
                        'semantic_component': semantic_component,
                        'topic_component': topic_component,
                        'specificity_boost': specificity_boost,
                        'final_score': final_score,
                        'matched_keywords': doc_data['matched_keywords'],
                        'semantic_weight': 0.55,
                        'specialized_weight': 0.25,
                        'topic_weight': 0.20,
                        'tfidf_word2vec_used': True,
                        'specialized_logic_used': True,
                        'program_info': spec_doc['program_info']
                    }
                })
            
            # Sort by final hybrid score
            hybrid_scored_results.sort(key=lambda x: x['relevance'], reverse=True)
            
            # Debug output
            print(f"✅ Top {min(top_k, len(hybrid_scored_results))} TRUE HYBRID results:")
            for i, doc in enumerate(hybrid_scored_results[:top_k]):
                debug = doc['_debug']
                print(f"   {i+1}. {doc['filename'][:50]}")
                print(f"       Final Score: {debug['final_score']:.3f}")
                print(f"       Semantic (55%): {debug['semantic_component']:.3f} (raw: {debug['semantic_similarity']:.3f})")
                print(f"       Specialized (25%): {debug['specialized_component']:.3f} (raw: {debug['specialized_score']:.3f})")
                print(f"       Topic (20%): {debug['topic_component']:.3f} (raw: {debug['topic_score']:.3f})")
                if debug['specificity_boost'] > 0:
                    print(f"       🎯 Specificity Boost: +{debug['specificity_boost']:.3f}")
                if debug.get('program_info'):
                    print(f"       📚 Program: {debug['program_info'].get('program_name', 'N/A')}")
                print(f"       Keywords: {debug['matched_keywords']}")
            
            return hybrid_scored_results[:top_k]
            
        except Exception as e:
            print(f"❌ Hybrid topic + semantic retrieval error: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to simple topic retrieval
            print("🔄 Falling back to simple topic retrieval...")
            return self.retrieve_documents_by_topic_keywords_simple(query, topic_id, top_k)

    def _calculate_query_document_specificity(self, query: str, metadata: dict) -> float:
        """
        Calculate specificity boost when query mentions specific terms that match document keywords/filename.
        This helps prioritize specific documents (e.g., international student docs when query mentions 'international').
        """
        import re
        
        query_lower = query.lower()
        filename = metadata.get('filename', '').lower()
        doc_keywords = metadata.get('keywords', '').lower()
        
        # Define specific terms that should boost matching documents
        specific_terms = {
            # Student types
            'international': ['international', 'foreign'],
            'transfer': ['transfer', 'transferee', 'shifter'],
            'scholar': ['scholar', 'scholarship'],
            'graduate': ['graduate', 'masters', 'phd', 'doctoral'],
            'undergraduate': ['undergraduate', 'bachelor'],
            
            # Program levels
            'first year': ['first year', '1st year', 'freshman'],
            'second year': ['second year', '2nd year', 'sophomore'],
            'third year': ['third year', '3rd year', 'junior'],
            'fourth year': ['fourth year', '4th year', 'senior'],
            
            # Document types
            'curriculum': ['curriculum', 'syllabus', 'course outline'],
            'fees': ['fees', 'tuition', 'cost', 'payment'],
            'visa': ['visa', 'immigration'],
            'requirements': ['requirements', 'documents needed'],
            
            # Specific processes
            'enrollment': ['enrollment', 'registration'],
            'admission': ['admission', 'application']
        }
        
        specificity_boost = 0.0
        matched_specific_terms = []
        
        # Check if query contains specific terms
        for term_category, term_variants in specific_terms.items():
            query_has_term = any(variant in query_lower for variant in term_variants)
            
            if query_has_term:
                # Check if document specifically addresses this term
                doc_matches_term = any(
                    variant in filename or variant in doc_keywords 
                    for variant in term_variants
                )
                
                if doc_matches_term:
                    # Boost score for documents that specifically match the query's specific terms
                    # Higher boost for student type specificity (international, transfer, etc.)
                    if term_category in ['international', 'transfer', 'scholar', 'graduate', 'undergraduate']:
                        boost_value = 0.20  # Higher boost for student type specificity
                    else:
                        boost_value = 0.15  # Standard boost for other specific terms
                    specificity_boost += boost_value
                    matched_specific_terms.append(term_category)
        
        # Additional boost for exact filename matches
        query_words = re.findall(r'\b\w+\b', query_lower)
        filename_words = re.findall(r'\b\w+\b', filename)
        
        # Count exact word matches between query and filename
        exact_matches = len(set(query_words) & set(filename_words))
        if exact_matches >= 2:  # At least 2 words match
            specificity_boost += 0.05 * exact_matches
            
        return min(specificity_boost, 0.3)  # Cap the boost to prevent over-boosting

    def _calculate_dynamic_top_k(self, topic_id: str, intent: str = None) -> int:
        """
        Dynamically calculate top_k based on intent and document count.
        
        Args:
            topic_id: The topic identifier (e.g., 'programs_courses', 'fees', 'admissions_enrollment')
            intent: Query intent ('overview', 'curriculum', 'subject_mapping', etc.) - optional
        
        Returns:
            int: Calculated top_k value
        """
        try:
            # For overview queries in programs_courses, get all program documents
            if topic_id == 'programs_courses' and intent == 'overview':
                try:
                    from .chroma_connection import ChromaService
                except ImportError:
                    from chroma_connection import ChromaService
                collection = ChromaService.get_client().get_or_create_collection(
                    name=self.chroma_collection_name
                )
                
                # Get all program-related documents (academic/curriculum type)
                all_docs = collection.get(
                    where={
                        "$and": [
                            {"source": "pdf_scrape"},
                            {"$or": [
                                {"document_type": "academic"},
                                {"document_type": "curriculum"}
                            ]}
                        ]
                    },
                    include=["metadatas"]
                )
                
                total_program_docs = len(all_docs.get('ids', []))
                # Add 50% buffer for future document additions
                safe_top_k = int(total_program_docs * 1.5) if total_program_docs > 0 else 100
                
                print(f"📊 Overview query - detected {total_program_docs} program documents, using dynamic top_k={safe_top_k}")
                return safe_top_k
            
            # For subject_mapping queries, also use higher top_k to find all programs with subject
            elif topic_id == 'programs_courses' and intent == 'subject_mapping':
                # Use similar logic as overview for subject mapping
                try:
                    from .chroma_connection import ChromaService
                except ImportError:
                    from chroma_connection import ChromaService
                collection = ChromaService.get_client().get_or_create_collection(
                    name=self.chroma_collection_name
                )
                
                all_docs = collection.get(
                    where={
                        "$and": [
                            {"source": "pdf_scrape"},
                            {"$or": [
                                {"document_type": "academic"},
                                {"document_type": "curriculum"}
                            ]}
                        ]
                    },
                    include=["metadatas"]
                )
                
                total_program_docs = len(all_docs.get('ids', []))
                safe_top_k = int(total_program_docs * 1.5) if total_program_docs > 0 else 100
                
                print(f"📊 Subject mapping query - detected {total_program_docs} program documents, using dynamic top_k={safe_top_k}")
                return safe_top_k
            
            # Default for specific queries (curriculum, duration, etc.)
            else:
                print(f"📊 Specific query (intent: {intent}) - using default top_k=3")
                return 3
                
        except Exception as e:
            print(f"⚠️ Error calculating dynamic top_k: {e}, falling back to default top_k=3")
            return 3

    def retrieve_documents_by_topic_specialized(self, query: str, topic_id: str, top_k: int = 2) -> List[Dict]:
        """
        Simplified dispatcher that uses hybrid retrieval only.
        Specialized handlers have been removed to rely on hybrid retrieval + LLM formatting.
        """
        print(f"🎯 Using hybrid-only retrieval for topic: {topic_id}")
        
        # Always use hybrid retrieval (no specialized handler routing)
        if hasattr(self, 'use_hybrid_topic_retrieval') and self.use_hybrid_topic_retrieval:
            print("✅ Using unified hybrid retrieval (JSON-enhanced TF-IDF + Word2Vec)")
            try:
                return self.retrieve_documents_by_topic_hybrid(query, topic_id, top_k)
            except Exception as e:
                print(f"❌ Hybrid retrieval failed: {e}")
                print("🔄 Falling back to simple topic filtering...")
                return self.retrieve_documents_by_topic_keywords_simple(query, topic_id, top_k)
        else:
            print(f"⚠️ Hybrid retrieval disabled, using simple method for {topic_id}")
            return self.retrieve_documents_by_topic_keywords_simple(query, topic_id, top_k)

    def retrieve_documents_hybrid(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        TWO-STAGE RETRIEVAL:
        1. Keyword search (fast, exact matches)
        2. Semantic search (fallback for fuzzy matches)
        Combines both for final ranking
        """
        import re
        
        print(f"�� Retrieving for: '{query}'")
        
        try:
            collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)
            
            # Extract query terms
            query_lower = query.lower()
            stop_words = {'what', 'is', 'the', 'a', 'an', 'for', 'in', 'on', 'at', 'to', 
                          'of', 'and', 'or', 'are', 'can', 'you', 'tell', 'me', 'about',
                          'how', 'when', 'where', 'who', 'which', 'do', 'does', 'i', 'my'}
            
            query_terms = set(re.findall(r'\b[a-z0-9]{2,}\b', query_lower)) - stop_words
            print(f"📝 Query terms: {query_terms}")
            
            # STAGE 1: Get ALL documents (no limit) to search keywords
            all_docs = collection.get(
                where={"source": "pdf_scrape"},
                include=["documents", "metadatas"]
            )
            
            all_ids = all_docs.get('ids', [])
            all_contents = all_docs.get('documents', [])
            all_metadatas = all_docs.get('metadatas', [])
            
            print(f"📚 Searching through {len(all_ids)} documents...")
            
            # KEYWORD SCORING for ALL documents
            keyword_candidates = []
            for i, (doc_id, content, metadata) in enumerate(zip(all_ids, all_contents, all_metadatas)):
                filename = metadata.get('filename', '')
                keywords = metadata.get('keywords', '')
                
                # Use improved keyword matching with synonyms
                keyword_score = self._improved_keyword_matching(query_terms, filename, keywords, content)
                
                # Only keep if it has SOME keyword match
                if keyword_score > 0:
                    keyword_candidates.append({
                        'id': doc_id,
                        'content': content,
                        'metadata': metadata,
                        'keyword_score': keyword_score
                    })
            
            print(f"�� Found {len(keyword_candidates)} documents with keyword matches")
            
            # STAGE 2: Get semantic embeddings for top keyword candidates
            q_emb = _embed_text(query)
            
            # Get embeddings for top 20 keyword candidates
            top_keyword_ids = [c['id'] for c in sorted(keyword_candidates, key=lambda x: x['keyword_score'], reverse=True)[:20]]
            
            # Also get top 20 semantic results
            semantic_results = collection.query(
                query_embeddings=[q_emb],
                n_results=20,
                include=["documents", "distances", "metadatas"],
                where={"source": "pdf_scrape"}
            )
            
            semantic_ids = semantic_results.get("ids", [[]])[0]
            semantic_distances = semantic_results.get("distances", [[]])[0]
            
            # Create semantic score lookup
            semantic_scores = {}
            for doc_id, distance in zip(semantic_ids, semantic_distances):
                semantic_scores[doc_id] = float(1.0 / (1.0 + distance))
            
            # COMBINE both approaches
            final_results = []
            seen_ids = set()
            
            # First, add keyword candidates with semantic scores
            for candidate in keyword_candidates:
                doc_id = candidate['id']
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                
                keyword_score = candidate['keyword_score']
                semantic_score = semantic_scores.get(doc_id, 0.3)  # Default low score if not in top semantic
                
                # Simple scoring without requiring specific match counts (since we changed the structure)
                if keyword_score > 0.5:
                    final_score = keyword_score * 0.8 + semantic_score * 0.2
                else:
                    final_score = keyword_score * 0.5 + semantic_score * 0.5
                
                final_results.append({
                    'id': doc_id,
                    'content': candidate['content'],
                    'relevance': final_score,
                    'folder': candidate['metadata'].get('folder_name', 'Unknown'),
                    'document_type': candidate['metadata'].get('document_type', 'other'),
                    'target_program': candidate['metadata'].get('target_program', 'all'),
                    'filename': candidate['metadata'].get('filename', ''),
                    'retrieval_strategy': 'keyword-first',
                    'hybrid_score': final_score,
                    '_debug': {
                        'semantic': semantic_score,
                        'keyword': keyword_score
                    }
                })
            
            # Sort by final score
            final_results.sort(key=lambda x: x['relevance'], reverse=True)
            
            # Debug output
            print(f"✅ Top {min(top_k, len(final_results))} results:")
            for i, doc in enumerate(final_results[:top_k]):
                debug = doc['_debug']
                print(f"   {i+1}. {doc['filename'][:70]}")
                print(f"       Score: {doc['relevance']:.3f} (sem={debug['semantic']:.3f}, kw={debug['keyword']:.3f})")
                print(f"       Matches: file={debug['filename_matches']}, kw={debug['keyword_matches']}, content={debug['content_matches']}")
            
            return final_results[:top_k]
            
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def retrieve_documents(self, query: str, top_k: int = 2) -> List[Dict]:
        """Main retrieval with intent analysis"""
        if self.use_chroma:
            # Use intent analysis to improve retrieval
            intent = self.analyze_query_intent(query)
            return self._retrieve_from_chroma(
                query, 
                top_k=top_k,
                document_type_filter=intent.get('document_type'),
                program_filter=intent.get('program_filter')
            )
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
            
            # Apply program acronym normalization before extracting terms
            normalized_query = self._normalize_program_acronyms(query)
            normalized_lower = normalized_query.lower()
            
            query_terms = set(re.findall(r'\b[a-z0-9]{2,}\b', normalized_lower))
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
                
                # IMPROVED KEYWORD MATCHING with synonyms
                filename = metadata.get('filename', '')
                keywords = metadata.get('keywords', '')
                
                keyword_score = self._improved_keyword_matching(query_terms, filename, keywords, content)
                
                # Program context boost - prioritize documents that match the program mentioned in query
                program_boost = self._calculate_program_context_boost(query, filename, keywords)
                
                # Combine: 50% semantic + 30% keyword + 20% program context
                final_score = (semantic_score * 0.5) + (keyword_score * 0.3) + (program_boost * 0.2)
                
                out.append({
                    "id": doc_id, 
                    "content": content, 
                    "relevance": final_score,
                    "folder": metadata.get("folder_name", "Unknown"),
                    "document_type": metadata.get("document_type", "other"),
                    "target_program": metadata.get("target_program", "all"),
                    "filename": metadata.get("filename", ""),
                    "_semantic": semantic_score,
                    "_keyword": keyword_score,
                    "_program_boost": program_boost
                })
            
            # Sort by combined score
            out.sort(key=lambda x: x['relevance'], reverse=True)
            
            print(f"📊 Top {min(top_k, len(out))} results:")
            for i, doc in enumerate(out[:top_k]):
                print(f"   {i+1}. {doc['filename'][:60]} | score={doc['relevance']:.3f} (sem={doc['_semantic']:.3f} + kw={doc['_keyword']:.3f} + prog={doc['_program_boost']:.3f})")
            
            return out[:top_k]
            
        except Exception as e:
            print(f"❌ Error querying Chroma: {e}")
            import traceback
            traceback.print_exc()
            return []

    def analyze_query_intent(self, query: str) -> Dict[str, str]:
        """Analyze query to determine appropriate filters using dynamic keyword matching"""
        query_lower = query.lower()
        
        # Get dynamic keyword mappings from database
        keyword_mappings = self._get_dynamic_keyword_mappings()
        
        # Document type detection using dynamic keywords
        document_type = None
        best_match_score = 0
        
        for doc_type, keywords in keyword_mappings.items():
            # Calculate match score for this document type
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            match_score = matches / len(keywords) if keywords else 0
            
            # Boost score for exact phrase matches
            exact_matches = sum(1 for keyword in keywords if len(keyword.split()) > 1 and keyword in query_lower)
            match_score += exact_matches * 0.5  # Boost for phrase matches
            
            if match_score > best_match_score:
                best_match_score = match_score
                document_type = doc_type
        
        # Special handling for admission office location queries
        if document_type == 'admission' and any(word in query_lower for word in ['office', 'location', 'where', 'contact', 'phone', 'email', 'address']):
            document_type = 'contact'
        
        # Program level detection (bachelor-focused, dynamic)
        program_filter = None
        
        # Focus on bachelor's/undergraduate programs only
        bachelor_indicators = [
            'undergraduate', 'bachelor', 'college', 'bs', 'ba',
            'bachelor of science', 'bachelor of arts', 'baccalaureate'
        ]
        
        # Dynamic program code detection from database
        bachelor_program_codes = self._get_bachelor_program_codes()
        
        # Check for bachelor indicators or program codes
        if (any(word in query_lower for word in bachelor_indicators) or 
            any(code in query_lower for code in bachelor_program_codes)):
            program_filter = 'undergraduate'
        
        # Since most documents are marked as 'all', default to 'all' for broader coverage
        # unless specifically requesting undergraduate-only content
        if program_filter is None:
            program_filter = 'all'
        
        return {
            'document_type': document_type,
            'program_filter': program_filter,
            'folder_filter': None,
            'match_confidence': best_match_score  # Add confidence score
        }
    
    def _get_dynamic_keyword_mappings(self) -> Dict[str, List[str]]:
        """Get keyword mappings dynamically from database documents"""
        # Cache the mappings to avoid repeated database queries
        if not hasattr(self, '_cached_keyword_mappings'):
            self._cached_keyword_mappings = self._build_keyword_mappings_from_db()
        
        return self._cached_keyword_mappings
    
    def _get_bachelor_program_codes(self) -> List[str]:
        """Get bachelor program codes dynamically from database and topic instructions"""
        # Cache the program codes to avoid repeated processing
        if not hasattr(self, '_cached_bachelor_codes'):
            self._cached_bachelor_codes = self._extract_bachelor_program_codes()
        
        return self._cached_bachelor_codes
    
    def _extract_bachelor_program_codes(self) -> List[str]:
        """Extract bachelor program codes from database keywords and topic instructions"""
        try:
            from .models import DocumentMetadata
            
            bachelor_codes = set()
            
            # Extract from document keywords that mention bachelor programs
            docs = DocumentMetadata.objects.exclude(keywords='').exclude(keywords__isnull=True)
            
            for doc in docs:
                keywords = [k.strip().lower() for k in doc.keywords.split(',') if k.strip()]
                
                # Look for bachelor program patterns (BS/BA + abbreviation)
                for keyword in keywords:
                    # Match patterns like "bs cs", "bscs", "ba eng", etc.
                    if keyword.startswith(('bs ', 'ba ', 'bscs', 'bsit', 'bsba', 'bs cs', 'bs it', 'bs ba')):
                        bachelor_codes.add(keyword)
                    # Also check for full program names that might indicate bachelor programs
                    elif any(indicator in keyword for indicator in ['computer science', 'information technology', 'business administration']):
                        # Extract potential abbreviations
                        if 'computer science' in keyword:
                            bachelor_codes.update(['bscs', 'bs cs', 'computer science'])
                        elif 'information technology' in keyword:
                            bachelor_codes.update(['bsit', 'bs it', 'information technology'])
                        elif 'business administration' in keyword:
                            bachelor_codes.update(['bsba', 'bs ba', 'business administration'])
            
            # Add common full program names for better detection
            full_program_names = [
                'computer science', 'information technology', 'business administration',
                'business management', 'accountancy', 'nursing', 'education',
                'engineering', 'architecture', 'mathematics', 'biology', 'chemistry',
                'psychology', 'economics', 'political science', 'sociology'
            ]
            bachelor_codes.update(full_program_names)
            
            # Add common bachelor program codes from the topic instructions
            # Based on the official program structure found in the codebase
            official_bachelor_codes = [
                # School of Arts & Sciences
                'ab eng', 'ab mc', 'ab ids', 'ab philo',  # Humanities & Letters
                'bs bio', 'bs chem', 'bs math', 'bs envi sci',  # Natural Sciences
                'bs is', 'bs it', 'bs cs', 'bs ds',  # Computer Studies
                'ab econ', 'ab polsci', 'ab psych', 'ab socio', 'ab anthro',  # Social Sciences
                
                # School of Business & Governance
                'bs a', 'bs ma',  # Accountancy
                'bs bm', 'bs entrep', 'bs fin', 'bs hrdm', 'bs mktg', 'bpm',  # Business Management
                
                # School of Education
                'bece', 'beed', 'bsed',
                
                # School of Engineering & Architecture
                'bs ae', 'bs arch', 'bs che', 'bs ce', 'bs comp eng', 'bs ee', 
                'bs electronics eng', 'bs ie', 'bs me', 'bs re',
                
                # School of Nursing
                'bs n'
            ]
            
            bachelor_codes.update(official_bachelor_codes)
            
            # Add common variations
            variations = set()
            for code in bachelor_codes:
                # Add space variations (e.g., "bscs" -> "bs cs")
                if code.startswith('bs') and len(code) > 2 and ' ' not in code:
                    variations.add(f"bs {code[2:]}")
                elif code.startswith('ab') and len(code) > 2 and ' ' not in code:
                    variations.add(f"ab {code[2:]}")
            
            bachelor_codes.update(variations)
            
            print(f"🎓 Extracted {len(bachelor_codes)} bachelor program codes")
            return list(bachelor_codes)
            
        except Exception as e:
            print(f"⚠️ Error extracting bachelor program codes: {e}")
            # Fallback to basic codes
            return ['bs', 'ba', 'bachelor', 'undergraduate']
    
    def _build_keyword_mappings_from_db(self) -> Dict[str, List[str]]:
        """Build keyword mappings from actual document keywords in database"""
        try:
            from .models import DocumentMetadata
            
            keyword_mappings = {}
            
            # Get all document types and their keywords
            docs = DocumentMetadata.objects.exclude(keywords='').exclude(keywords__isnull=True)
            
            for doc in docs:
                doc_type = doc.document_type
                if doc_type not in keyword_mappings:
                    keyword_mappings[doc_type] = set()
                
                # Extract keywords from document
                keywords = [k.strip().lower() for k in doc.keywords.split(',') if k.strip()]
                keyword_mappings[doc_type].update(keywords)
            
            # Convert sets to lists and add common query variations
            for doc_type in keyword_mappings:
                keywords = list(keyword_mappings[doc_type])
                
                # Add common variations and synonyms
                expanded_keywords = set(keywords)
                
                for keyword in keywords:
                    # Add plural/singular variations
                    if keyword.endswith('s') and len(keyword) > 3:
                        expanded_keywords.add(keyword[:-1])
                    elif not keyword.endswith('s'):
                        expanded_keywords.add(keyword + 's')
                    
                    # Add common query words for each type
                    if doc_type == 'policy':
                        expanded_keywords.update(['rule', 'rules', 'regulation', 'regulations'])
                    elif doc_type == 'academic':
                        expanded_keywords.update(['program', 'course', 'degree', 'major'])
                    elif doc_type == 'fees':
                        expanded_keywords.update(['cost', 'price', 'amount', 'charge'])
                    elif doc_type == 'contact':
                        expanded_keywords.update(['phone', 'email', 'address', 'location'])
                
                keyword_mappings[doc_type] = list(expanded_keywords)
            
            print(f"🔍 Built dynamic keyword mappings for {len(keyword_mappings)} document types")
            return keyword_mappings
            
        except Exception as e:
            print(f"⚠️ Error building keyword mappings: {e}")
            # Fallback to basic mappings
            return {
                'policy': ['policy', 'grading', 'grade', 'retention', 'dismissal'],
                'academic': ['program', 'course', 'degree', 'curriculum'],
                'fees': ['fee', 'tuition', 'cost', 'payment'],
                'contact': ['contact', 'office', 'phone', 'email'],
                'admission': ['admission', 'apply', 'requirement'],
                'enrollment': ['enrollment', 'registration', 'process'],
                'scholarship': ['scholarship', 'financial aid', 'grant']
            }

    def _expand_query_terms_with_synonyms(self, query_terms: set) -> set:
        """Minimal term expansion - just handle plural/singular"""
        expanded_terms = set(query_terms)
        
        for term in query_terms:
            # Simple plural/singular handling
            if term.endswith('s') and len(term) > 3:
                expanded_terms.add(term[:-1])
            elif not term.endswith('s'):
                expanded_terms.add(term + 's')
        
        return expanded_terms

    def _improved_keyword_matching(self, query_terms: set, filename: str, keywords: str, content: str) -> float:
        """Improved keyword matching with flexible term matching"""
        filename_lower = filename.lower()
        keywords_lower = keywords.lower()
        content_lower = content.lower()
        
        matches = 0
        total_possible_matches = len(query_terms) if query_terms else 1
        
        for term in query_terms:
            term_score = 0
            
            # Exact word boundary matching (highest score)
            import re
            term_pattern = r'\b' + re.escape(term) + r'\b'
            
            if re.search(term_pattern, filename_lower):
                term_score = max(term_score, 1.5)
            elif re.search(term_pattern, keywords_lower):
                term_score = max(term_score, 1.3)
            elif re.search(term_pattern, content_lower):
                term_score = max(term_score, 1.0)
            
            # Flexible matching for common variations
            elif term in filename_lower:
                term_score = max(term_score, 1.2)
            elif term in keywords_lower:
                term_score = max(term_score, 1.0)
            elif term in content_lower:
                term_score = max(term_score, 0.8)
            
            # Handle plural/singular automatically
            elif term.endswith('s') and term[:-1] in keywords_lower:
                term_score = max(term_score, 0.9)  # Plural -> singular match
            elif not term.endswith('s') and (term + 's') in keywords_lower:
                term_score = max(term_score, 0.9)  # Singular -> plural match
            
            matches += term_score
        
        keyword_score = min(matches / total_possible_matches, 1.0)
        return keyword_score
    
    def _calculate_program_context_boost(self, query: str, filename: str, keywords: str) -> float:
        """Calculate boost score for documents that match the program context in the query"""
        query_lower = query.lower()
        filename_lower = filename.lower()
        keywords_lower = keywords.lower()
        
        # Get program patterns for matching
        program_patterns = self._get_program_patterns()
        
        boost_score = 0.0
        
        # Check if any program is mentioned in the query
        for program_name, patterns in program_patterns.items():
            program_mentioned = False
            
            # Check if this program is mentioned in the query
            import re
            for pattern in patterns:
                # Use word boundaries for short patterns to avoid false matches
                if len(pattern) <= 3:
                    # Short patterns like 'cs', 'it' need word boundaries
                    if re.search(r'\b' + re.escape(pattern) + r'\b', query_lower):
                        program_mentioned = True
                        break
                elif len(pattern.split()) > 1:
                    # Multi-word patterns like 'computer science' need exact phrase match
                    if pattern in query_lower:
                        # Additional check: ensure it's not part of a larger phrase
                        pattern_words = set(pattern.split())
                        query_words = set(query_lower.split())
                        if pattern_words.issubset(query_words):
                            program_mentioned = True
                            break
                else:
                    # Single word patterns need word boundaries to avoid substring matches
                    if re.search(r'\b' + re.escape(pattern) + r'\b', query_lower):
                        program_mentioned = True
                        break
            
            if program_mentioned:
                # Check if this document is about the same program
                document_matches_program = False
                
                # Check filename and keywords for program match
                for pattern in patterns:
                    if pattern in filename_lower or pattern in keywords_lower:
                        document_matches_program = True
                        break
                
                if document_matches_program:
                    # Strong boost for exact program match
                    boost_score = 1.0
                    break
                else:
                    # Check for related programs (e.g., all engineering programs)
                    if 'engineering' in program_name and 'engineering' in filename_lower:
                        boost_score = max(boost_score, 0.3)  # Moderate boost for related programs
        
        return boost_score

    def process_query_with_intent_analysis(self, query: str, correct_spelling: bool = True, 
                                          max_tokens: int = 1024, stream: bool = True, 
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
            # Initialize curriculum state when program is set
            if 'curriculum_state' not in self.session_state:
                self.session_state['curriculum_state'] = {
                    'current_program': current_program,
                    'last_displayed_year': None,
                    'parsed_curriculum': {},
                    'available_years': []
                }
            else:
                self.session_state['curriculum_state']['current_program'] = current_program
        
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

    def process_query(self, query: str, correct_spelling: bool = True, max_tokens: int = 1024,
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
                # Check if topic exists in database
                topic_info = get_topic_info(topic_id)
                if not topic_info:
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
                
                # Special welcome message for programs_courses topic
                if topic_id == 'programs_courses':
                    welcome_message = f"""Great! You've selected **{topic_info['label']}**. {topic_info['description']}

If you want to see the list of programs, click on the buttons below per school, or if you want to ask for specific curricula, just type it in the chatbox below."""
                    
                    # Create school buttons dynamically from JSON config
                    school_buttons = self._generate_school_buttons()
                    
                    button_configs = get_button_configs()
                    # Add school buttons to the existing action buttons
                    all_buttons = button_configs['topic_conversation']['buttons'] + school_buttons
                    
                    return {
                        'response': welcome_message,
                        'state': CONVERSATION_STATES['TOPIC_CONVERSATION'],
                        'buttons': all_buttons,
                        'input_enabled': button_configs['topic_conversation']['input_enabled'],
                        'current_topic': topic_id,
                    }
                else:
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
                
                # Handle school button clicks
                elif action_data.startswith('school_'):
                    school_programs = self._get_school_programs(action_data)
                    
                    # Create school buttons again for continued interaction
                    school_buttons = self._generate_school_buttons()
                    
                    button_configs = get_button_configs()
                    all_buttons = button_configs['topic_conversation']['buttons'] + school_buttons
                    
                    return {
                        'response': school_programs,
                        'state': CONVERSATION_STATES['TOPIC_CONVERSATION'],
                        'buttons': all_buttons,
                        'input_enabled': button_configs['topic_conversation']['input_enabled'],
                        'current_topic': current_topic
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
                        
                        # Add to dialogue history for conversation context
                        self.add_to_history(user_input, response)
                        
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
                    
                    # Add to dialogue history for conversation context
                    self.add_to_history(user_input, response)
                    
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
                        
                        # Add to dialogue history for conversation context
                        self.add_to_history(user_input, response)
                        
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
- **FOCUS**: Cover admission requirements, required documents, processes, and procedures specific to the identified student type

=== LINK HANDLING FOR ADMISSIONS ===
- **NO LINKS RULE**: Do NOT mention, suggest, or provide ANY links or URLs
- **NO FABRICATION**: NEVER create, invent, or fabricate URLs
- **NO LINK PHRASES**: Do NOT use phrases like "View Curriculum", "click here", "for more information", or any link-related text
- **NO EXTERNAL REFERENCES**: Do NOT mention external resources, websites, or links
- **FOCUS ON CONTENT**: Provide information directly from the source documents without referencing external links
- **NO END LINKS**: Do NOT end responses with link suggestions or "for more information" statements"""

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
- **YEAR-BASED DISPLAY**: Curriculum queries show subjects for the requested year level
- **DEFAULT YEAR**: If no year is specified, default to Year 1 curriculum
- **FOLLOW-UP SUPPORT**: Users can ask "what about 2nd year" or "show me 3rd year" to navigate through different years
- **MULTI-PROGRAM QUERIES**: If multiple programs are mentioned (e.g., "compare CS and IT curriculum" or "CS and IT 3rd year"), format as a comparison showing both curricula side-by-side with clear separation between programs
- **FORMAT**: 
  * Show program name/acronym
  * Show year level
  * List subjects organized by semester (First Semester, Second Semester, Summer if applicable)
  * **Semester headers**: Format semester names in bold with total Credit Units (e.g., **First Semester (21.0 CU)**, **Second Semester (21.0 CU)**)
  * **Empty semesters**: If a semester has no courses, DO NOT show that semester header at all - omit it completely (no "No courses" message)
  * Include course codes, titles, and credits for each course
  * Include curriculum PDF link if available in the document
- **SESSION CONTINUITY**: Remember which program and year was last displayed for follow-up queries

=== RESPONSE FORMAT ===
- For availability: Start with clear confirmation based on official documents
- For program lists: Use the exact structure from the official document
- For curriculum: Combine program confirmation + curriculum details
- Always cite sources when providing program information

=== AMBIGUOUS PROGRAM RESPONSES ===
- When multiple programs match a query (e.g., "AB IDS" matches multiple IDS programs):
  * Start with: "I found multiple programs that match your query. Here are the available options:"
  * Format each program on a separate line with proper line breaks
  * Use this exact format:
    ```
    School of [Name] - [Cluster]:
    • [Program Code] - [Full Program Name]
    • [Program Code] - [Full Program Name]
    ```
  * Group programs by school and cluster
  * End with: "Could you please specify which program you're interested in? You can ask about any of these [X] programs by name."

=== LINK HANDLING ===
- **ALLOW LINKS**: Include links if they are explicitly present in the source document content
- **NO FABRICATION**: NEVER create, invent, or fabricate URLs
- **HYPERLINKS**: If URLs exist, format them as clickable hyperlinks using Markdown syntax: [link text](URL)
- **CRITICAL**: Use the EXACT URL from the document content - DO NOT modify, autocorrect, or change ANY part of the URL
- **NO CORRECTIONS**: Do NOT fix typos in URLs, do NOT change "Technolgy" to "Technology", do NOT modify any part of the original URL
- **PRESERVE ORIGINAL**: Copy the URL character-for-character exactly as it appears in the source document
- **FORMAT**: If URLs exist, use "For more information about [topic], head to this link: [link text](EXACT_URL_FROM_DOCUMENT)"
- **NO LINKS RULE**: If no URLs are present in the source documents, do NOT mention links at all

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

**School of Nursing**: BS N

=== SUBJECT MAPPING QUERIES ===
- **FLEXIBLE HANDLING**: Handle both general and program-specific queries using different response formats

- **GENERAL QUERIES** ("what programs have [subject]", "which programs offer [subject]"):
  * **RESPONSE FORMAT**: Organized by School and Cluster with bullet points
  * **FORMAT EXAMPLE**: 
    ```
    Programs with Math:
    
    School of Arts and Sciences
    • Natural Sciences & Mathematics (Cluster)
      • Bachelor of Science in Mathematics
      • Bachelor of Science in Environmental Science
      • Bachelor of Science in Computer Science
      • Bachelor of Science in Data Science
    
    School of Engineering & Architecture
    • Engineering (Cluster)
      • Bachelor of Science in Civil Engineering
      • Bachelor of Science in Electrical Engineering
      • Bachelor of Science in Mechanical Engineering
    
    School of Education
    • Education (Cluster)
      • Bachelor of Secondary Education Major in Mathematics
      • Bachelor of Elementary Education
    ```
  * **DO NOT INCLUDE**: Detailed curriculum, course codes, or sample courses
  * **ORGANIZE**: Group by School → Cluster → Programs using bullet points
  * **BE COMPREHENSIVE**: Include ALL programs found in the documents, not just top 3

- **PROGRAM-SPECIFIC QUERIES** ("what [subject] subjects are in [program]", "show me programming courses in CS"):
  * **CRITICAL**: This is NOT a curriculum query - do NOT use curriculum formatting rules
  * **SEARCH SCOPE**: Search across ALL YEARS of the program's curriculum (Year 1, 2, 3, 4)
  * **TITLE FORMAT**: Use "**[Subject Type] Subjects in [Program Name]**" with bold formatting - NEVER use "Year 1" in title
  * **YEAR FORMATTING**: Use bold formatting for years (e.g., "**Year 1:**", "**Year 2:**")
  * **SEMESTER FORMATTING**: Use bold formatting for semesters (e.g., "**First Semester:**", "**Second Semester:**", "**Summer Semester:**")
  * **OVERRIDE DEFAULT**: For subject mapping queries, DO NOT default to Year 1 - show ALL years
  * **INCLUDE**: Course codes, titles, and year/semester information from ALL years
  * **ORGANIZE**: Group by year and semester for better readability, but show ALL years
  * **BE COMPREHENSIVE**: Show all relevant subjects across entire program duration
  * **CRITICAL**: This overrides the "default to Year 1" rule for curriculum queries - subject mapping must show ALL years
  * **IGNORE**: Any curriculum formatting rules that limit to Year 1

- **EXCLUDE ALWAYS**: General program listings, school/cluster breakdowns, unrelated program information
- **TRUST RETRIEVAL**: The documents provided already contain programs/subjects relevant to the query
- **SEMANTIC FLEXIBILITY**: Use semantic understanding for subject matching (math=mathematics, OJT=practicum/internship)"""

        elif topic_id == 'fees':
            return """TOPIC-SPECIFIC INSTRUCTIONS FOR FEES:

=== FEE INFORMATION HANDLING ===
- **BASE RESPONSES** on the specific COURSE/PROGRAM mentioned by the user
- **MATCH**: Provide fee information specific to the program the user asked about
- **INCLUDE**: Tuition fees, miscellaneous fees, payment schedules, installment options for that specific program
- **DIFFERENTIATE**: Different programs may have different fee structures
- **SCOPE**: Undergraduate program fees only

=== LINK HANDLING FOR FEES ===
- **NO LINKS RULE**: Do NOT mention, suggest, or provide ANY links or URLs
- **NO FABRICATION**: NEVER create, invent, or fabricate URLs
- **NO LINK PHRASES**: Do NOT use phrases like "View Curriculum", "click here", "for more information", or any link-related text
- **NO EXTERNAL REFERENCES**: Do NOT mention external resources, websites, or links
- **FOCUS ON CONTENT**: Provide information directly from the source documents without referencing external links
- **NO END LINKS**: Do NOT end responses with link suggestions or "for more information" statements"""

        else:
            # Generic instructions for any other topics
            return """TOPIC-SPECIFIC INSTRUCTIONS:
- Focus on answering questions within this topic area
- Use only information from the provided context documents"""
    
    def _load_normalization_config(self):
        """Load and cache the normalization configuration from JSON file"""
        if self._normalization_config is not None:
            return self._normalization_config
        
        try:
            import json
            with open(self._config_file_path, 'r', encoding='utf-8') as f:
                self._normalization_config = json.load(f)
            print(f"[CONFIG] Loaded normalization config from {self._config_file_path}")
            return self._normalization_config
        except FileNotFoundError:
            print(f"[ERROR] Normalization config file not found: {self._config_file_path}")
            # Return empty config as fallback
            return {
                "program_abbreviations": {},
                "context_patterns": {"problematic": [], "program_keywords": [], "strong_program": []},
                "common_english_words": []
            }
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON in normalization config: {e}")
            return {
                "program_abbreviations": {},
                "context_patterns": {"problematic": [], "program_keywords": [], "strong_program": []},
                "common_english_words": []
            }
    
    def _reload_normalization_config(self):
        """Force reload of normalization configuration"""
        self._normalization_config = None
        return self._load_normalization_config()
    
    def _get_schools_from_config(self) -> list:
        """Extract unique schools from JSON config"""
        config = self._load_normalization_config()
        schools = set()
        
        # Extract from program_abbreviations
        for school_section, programs_data in config.get("program_abbreviations", {}).items():
            if school_section.startswith('_'):  # Skip metadata
                continue
                
            for abbrev_key, abbrev_data in programs_data.items():
                if isinstance(abbrev_data, dict) and 'school' in abbrev_data:
                    schools.add(abbrev_data['school'])
        
        # Also extract from school_abbreviations and school_keywords
        school_abbrevs = config.get("school_abbreviations", {})
        school_keywords = config.get("school_keywords", {})
        
        for abbrev, school_name in school_abbrevs.items():
            if not abbrev.startswith('_'):
                schools.add(school_name)
        
        for keyword, school_name in school_keywords.items():
            if not keyword.startswith('_'):
                schools.add(school_name)
        
        return sorted(list(schools))
    
    def _get_programs_by_school_from_config(self, school_name: str) -> dict:
        """Get programs organized by cluster for a specific school from JSON config"""
        config = self._load_normalization_config()
        school_programs = {}
        
        # Extract programs for the specified school
        for school_section, programs_data in config.get("program_abbreviations", {}).items():
            if school_section.startswith('_'):  # Skip metadata
                continue
                
            for abbrev_key, abbrev_data in programs_data.items():
                if (isinstance(abbrev_data, dict) and 
                    abbrev_data.get('school') == school_name and 
                    'full_name' in abbrev_data):
                    
                    cluster = abbrev_data.get('cluster', 'Other')
                    if cluster not in school_programs:
                        school_programs[cluster] = []
                    
                    school_programs[cluster].append({
                        'abbreviation': abbrev_data['full_name'],
                        'description': abbrev_data.get('description', ''),
                        'priority': abbrev_data.get('priority', '')
                    })
        
        return school_programs
    
    def _generate_school_buttons(self) -> list:
        """Generate school buttons dynamically from JSON config"""
        schools = self._get_schools_from_config()
        school_buttons = []
        
        # Create mapping from school names to button IDs
        school_id_mapping = {
            'School of Arts & Sciences': 'school_arts_sciences',
            'School of Business & Governance': 'school_business_governance',
            'School of Education': 'school_education',
            'School of Engineering & Architecture': 'school_engineering_architecture',
            'School of Nursing': 'school_nursing'
        }
        
        for school in schools:
            if school in school_id_mapping:
                school_buttons.append({
                    'id': school_id_mapping[school],
                    'label': school,
                    'type': 'school'
                })
        
        return school_buttons
    
    def _get_program_abbreviations(self):
        """Get all program abbreviations from config"""
        config = self._load_normalization_config()
        abbreviations = {}
        
        # Flatten all school sections into a single dict
        for school_section, programs in config["program_abbreviations"].items():
            if isinstance(programs, dict) and not school_section.startswith("_"):
                abbreviations.update(programs)
        
        return abbreviations
    
    def _get_context_patterns(self):
        """Get context patterns from config"""
        config = self._load_normalization_config()
        return config.get("context_patterns", {})
    
    def _extract_keywords_from_json(self, topic_id: str) -> list:
        """Extract all relevant keywords from JSON config for programs topic"""
        if topic_id != 'programs_courses':
            # For other topics, use existing topic keywords
            from .topics import get_topic_keywords
            return get_topic_keywords(topic_id) or []
        
        print(f"📝 Extracting keywords from JSON config for {topic_id}")
        
        try:
            config = self._load_normalization_config()
            keywords = set()
            
            # Extract all program-related keywords
            program_abbrevs = config.get('program_abbreviations', {})
            for school_section, programs in program_abbrevs.items():
                if school_section.startswith('_'):  # Skip metadata
                    continue
                    
                for abbrev_key, abbrev_data in programs.items():
                    if isinstance(abbrev_data, dict):
                        # Add abbreviation key
                        keywords.add(abbrev_key.lower())
                        
                        # Add full name
                        full_name = abbrev_data.get('full_name', '')
                        if full_name:
                            keywords.add(full_name.lower())
                            # Add individual words from full name
                            keywords.update(word.lower() for word in full_name.split() if len(word) > 1)
                        
                        # Add description terms
                        description = abbrev_data.get('description', '')
                        if description:
                            # Extract key terms from description
                            desc_words = description.lower().split()
                            for word in desc_words:
                                if len(word) > 3 and word not in ['bachelor', 'science', 'arts', 'major', 'minor']:
                                    keywords.add(word)
            
            # Add cluster keywords
            cluster_keywords = config.get('cluster_keywords', {})
            for keyword, cluster_name in cluster_keywords.items():
                keywords.add(keyword.lower())
                keywords.add(cluster_name.lower())
            
            # Add school keywords and abbreviations
            school_keywords = config.get('school_keywords', {})
            for keyword, school_name in school_keywords.items():
                keywords.add(keyword.lower())
                keywords.add(school_name.lower())
            
            school_abbreviations = config.get('school_abbreviations', {})
            for abbrev, school_name in school_abbreviations.items():
                keywords.add(abbrev.lower())
                keywords.add(school_name.lower())
            
            # Add common program-related terms
            program_terms = ['program', 'course', 'curriculum', 'degree', 'bachelor', 'undergraduate', 'graduate']
            keywords.update(program_terms)
            
            result = list(keywords)
            print(f"📝 Extracted {len(result)} keywords from JSON config")
            return result
            
        except Exception as e:
            print(f"❌ Error extracting keywords from JSON: {e}")
            # Fallback to topic keywords
            from .topics import get_topic_keywords
            return get_topic_keywords(topic_id) or []
    
    def _get_all_programs_from_json(self) -> dict:
        """Extract all programs from JSON (all schools, all clusters)"""
        try:
            config = self._load_normalization_config()
            programs = {}
            
            program_abbrevs = config.get('program_abbreviations', {})
            for school_section, school_programs in program_abbrevs.items():
                if school_section.startswith('_'):  # Skip metadata
                    continue
                    
                for abbrev_key, abbrev_data in school_programs.items():
                    if isinstance(abbrev_data, dict):
                        full_name = abbrev_data.get('full_name', abbrev_key.upper())
                        
                        # Use full_name as the key for consistency
                        if full_name not in programs:
                            programs[full_name] = {
                                'abbrev': abbrev_key,
                                'full_name': full_name,
                                'school': abbrev_data.get('school', ''),
                                'cluster': abbrev_data.get('cluster', ''),
                                'description': abbrev_data.get('description', ''),
                                'variations': [abbrev_key.lower(), full_name.lower()]
                            }
                            
                            # Add description terms as variations
                            description = abbrev_data.get('description', '')
                            if description and 'bachelor of' in description.lower():
                                # Extract field name from description
                                field_name = description.lower()
                                for prefix in ['bachelor of science in ', 'bachelor of arts in ', 'bachelor of ']:
                                    if prefix in field_name:
                                        field_name = field_name.replace(prefix, '').strip()
                                        if field_name:
                                            programs[full_name]['variations'].append(field_name)
                                        break
            
            print(f"📚 Extracted {len(programs)} programs from JSON config")
            return programs
            
        except Exception as e:
            print(f"❌ Error extracting programs from JSON: {e}")
            return {}
    
    def _get_programs_by_cluster_from_json(self, cluster_name: str) -> list:
        """Look up cluster and extract all programs with matching cluster field"""
        try:
            config = self._load_normalization_config()
            
            # First, normalize cluster name using cluster_keywords
            cluster_keywords = config.get('cluster_keywords', {})
            normalized_cluster = None
            
            # Check if cluster_name matches any keyword
            for keyword, official_cluster in cluster_keywords.items():
                if keyword.lower() == cluster_name.lower():
                    normalized_cluster = official_cluster
                    break
            
            # If not found in keywords, use as-is
            if not normalized_cluster:
                normalized_cluster = cluster_name
            
            print(f"🎯 Looking for programs in cluster: '{normalized_cluster}'")
            
            # Extract programs with matching cluster
            programs = []
            program_abbrevs = config.get('program_abbreviations', {})
            
            for school_section, school_programs in program_abbrevs.items():
                if school_section.startswith('_'):  # Skip metadata
                    continue
                    
                for abbrev_key, abbrev_data in school_programs.items():
                    if isinstance(abbrev_data, dict):
                        program_cluster = abbrev_data.get('cluster', '')
                        if program_cluster.lower() == normalized_cluster.lower():
                            full_name = abbrev_data.get('full_name', abbrev_key.upper())
                            programs.append({
                                'abbrev': abbrev_key,
                                'full_name': full_name,
                                'school': abbrev_data.get('school', ''),
                                'cluster': program_cluster,
                                'description': abbrev_data.get('description', '')
                            })
            
            print(f"📚 Found {len(programs)} programs in cluster '{normalized_cluster}'")
            return programs
            
        except Exception as e:
            print(f"❌ Error getting programs by cluster: {e}")
            return []
    
    def _get_programs_by_school_from_json(self, school_name_or_abbrev: str) -> list:
        """Check school_keywords and school_abbreviations to map to full school name, then extract programs"""
        try:
            config = self._load_normalization_config()
            
            # Normalize school name
            school_keywords = config.get('school_keywords', {})
            school_abbreviations = config.get('school_abbreviations', {})
            
            normalized_school = None
            
            # Check school_keywords first
            for keyword, official_school in school_keywords.items():
                if keyword.lower() == school_name_or_abbrev.lower():
                    normalized_school = official_school
                    break
            
            # Check school_abbreviations if not found
            if not normalized_school:
                for abbrev, official_school in school_abbreviations.items():
                    if abbrev.lower() == school_name_or_abbrev.lower():
                        normalized_school = official_school
                        break
            
            # If still not found, use as-is
            if not normalized_school:
                normalized_school = school_name_or_abbrev
            
            print(f"🏫 Looking for programs in school: '{normalized_school}'")
            
            # Extract programs with matching school
            programs = []
            program_abbrevs = config.get('program_abbreviations', {})
            
            for school_section, school_programs in program_abbrevs.items():
                if school_section.startswith('_'):  # Skip metadata
                    continue
                    
                for abbrev_key, abbrev_data in school_programs.items():
                    if isinstance(abbrev_data, dict):
                        program_school = abbrev_data.get('school', '')
                        if program_school.lower() == normalized_school.lower():
                            full_name = abbrev_data.get('full_name', abbrev_key.upper())
                            programs.append({
                                'abbrev': abbrev_key,
                                'full_name': full_name,
                                'school': program_school,
                                'cluster': abbrev_data.get('cluster', ''),
                                'description': abbrev_data.get('description', '')
                            })
            
            print(f"📚 Found {len(programs)} programs in school '{normalized_school}'")
            return programs
            
        except Exception as e:
            print(f"❌ Error getting programs by school: {e}")
            return []
    
    def _detect_cluster_or_school_in_query(self, query: str) -> dict:
        """Detect if query mentions cluster, school, or is general"""
        try:
            config = self._load_normalization_config()
            query_lower = query.lower()
            
            # Check for cluster keywords
            cluster_keywords = config.get('cluster_keywords', {})
            for keyword, cluster_name in cluster_keywords.items():
                if keyword.lower() in query_lower:
                    return {'type': 'cluster', 'value': cluster_name}
            
            # Check for school keywords
            school_keywords = config.get('school_keywords', {})
            for keyword, school_name in school_keywords.items():
                if keyword.lower() in query_lower:
                    return {'type': 'school', 'value': school_name}
            
            # Check for school abbreviations
            school_abbreviations = config.get('school_abbreviations', {})
            for abbrev, school_name in school_abbreviations.items():
                if abbrev.lower() in query_lower:
                    return {'type': 'school', 'value': school_name}
            
            # Check for general query patterns
            general_patterns = [
                r'\ball\s+programs?\b',
                r'\blist\s+programs?\b',
                r'\bprograms?\s+in\s+addu\b',
                r'\bprograms?\s+offered\b',
                r'\bwhat\s+programs?\s+does\s+addu\s+offer\b',
                r'\bavailable\s+programs?\b'
            ]
            
            import re
            for pattern in general_patterns:
                if re.search(pattern, query_lower):
                    return {'type': 'general', 'value': None}
            
            # Default to general if no specific cluster/school detected
            return {'type': 'general', 'value': None}
            
        except Exception as e:
            print(f"❌ Error detecting cluster/school in query: {e}")
            return {'type': 'general', 'value': None}

    def _detect_year_semester_query(self, query: str) -> dict:
        """Detect if query is asking about program years/semesters"""
        import re
        
        query_lower = query.lower()
        
        # Year range patterns (e.g., "3-4 years", "3 to 4 years")
        year_range_patterns = [
            r'(\d+)\s*[-–—]\s*(\d+)\s+years?',
            r'(\d+)\s+to\s+(\d+)\s+years?',
            r'(\d+)\s*[-–—]\s*(\d+)\s*year',
            r'between\s+(\d+)\s+and\s+(\d+)\s+years?'
        ]
        
        # Single year patterns (e.g., "4 years", "4-year program")
        single_year_patterns = [
            r'(\d+)\s+years?',
            r'(\d+)[-–—]year',
            r'are\s+(\d+)\s+years?',
            r'have\s+(\d+)\s+years?',
            r'that\s+are\s+(\d+)\s+years?'
        ]
        
        # Semester patterns (flexible matching for various phrasings)
        semester_patterns = [
            r'no\s+(summer|first|second)\s+semester',           # "no summer semester"
            r'has\s+no\s+(summer|first|second)\s+semester',     # "has no summer semester"
            r'have\s+no\s+(summer|first|second)\s+semester',    # "have no summer semester"
            r'that\s+has\s+no\s+(summer|first|second)\s+semester',  # "that has no summer semester"
            r'have\s+(summer|first|second)\s+semester',         # "have summer semester"
            r'has\s+(summer|first|second)\s+semester',          # "has summer semester"
            r'with\s+(summer|first|second)\s+semester',         # "with summer semester"
            r'without\s+(summer|first|second)\s+semester',      # "without summer semester"
            r'(summer|first|second)\s+semester\s+is\s+not',     # "summer semester is not"
            r'don\'?t\s+have\s+(summer|first|second)\s+semester', # "don't have summer semester"
        ]
        
        result = {
            'is_year_semester_query': False,
            'query_type': None,
            'min_years': None,
            'max_years': None,
            'exact_years': None,
            'semester_requirement': None,
            'semester_type': None,
            'is_negative': False
        }
        
        # Check for year range queries
        for pattern in year_range_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result['is_year_semester_query'] = True
                result['query_type'] = 'year_range'
                result['min_years'] = int(match.group(1))
                result['max_years'] = int(match.group(2))
                print(f"🎯 Detected year range query: {result['min_years']}-{result['max_years']} years")
                return result
        
        # Check for single year queries
        for pattern in single_year_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result['is_year_semester_query'] = True
                result['query_type'] = 'exact_years'
                result['exact_years'] = int(match.group(1))
                print(f"🎯 Detected exact year query: {result['exact_years']} years")
                return result
        
        # Check for semester queries
        for pattern in semester_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result['is_year_semester_query'] = True
                result['query_type'] = 'semester'
                result['semester_type'] = match.group(1)
                # More accurate negative detection
                result['is_negative'] = ('no' in pattern or 'without' in pattern or "don't" in pattern or 
                                       'no ' in query_lower or 'without ' in query_lower or "don't " in query_lower)
                result['semester_requirement'] = 'exclude' if result['is_negative'] else 'include'
                print(f"🎯 Detected semester query: {result['semester_requirement']} {result['semester_type']} semester")
                return result
        
        return result

    def _detect_duration_query(self, query: str) -> dict:
        """Detect if query is asking about program duration only"""
        import re
        
        query_lower = query.lower()
        
        # Duration query patterns
        duration_patterns = [
            r'how many years (?:for|is) ([A-Za-z\s]+)',          # "how many years for BS CS"
            r'how many years (?:does|is) ([A-Za-z\s]+)',        # "how many years does BS CS have"
            r'how long is (?:the )?([A-Za-z\s]+) program',      # "how long is the BS CS program"
            r'duration of (?:the )?([A-Za-z\s]+)',              # "duration of BS CS"
            r'what is the duration of ([A-Za-z\s]+)',           # "what is the duration of BS CS"
            r'how many years to complete ([A-Za-z\s]+)',        # "how many years to complete BS CS"
            r'length of (?:the )?([A-Za-z\s]+) program',        # "length of the BS CS program"
        ]
        
        result = {
            'is_duration_query': False,
            'program': None
        }
        
        for pattern in duration_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result['is_duration_query'] = True
                result['program'] = match.group(1).strip()
                print(f"🎯 Detected duration query for program: '{result['program']}'")
                return result
        
        return result

    def _count_years_and_semesters_from_curriculum(self, curriculum_content: str) -> dict:
        """Count total years and semester types from curriculum content"""
        try:
            # Parse curriculum using existing method
            curriculum_structure = self._parse_curriculum_by_years(curriculum_content)
            
            if not curriculum_structure:
                return {'total_years': 0, 'semesters': [], 'has_summer': False}
            
            # Count years
            total_years = max(curriculum_structure.keys()) if curriculum_structure else 0
            
            # Collect all semester types across all years
            all_semesters = set()
            for year_data in curriculum_structure.values():
                all_semesters.update(year_data.keys())
            
            # Check for summer semester
            has_summer = 'summer' in all_semesters
            
            result = {
                'total_years': total_years,
                'semesters': sorted(list(all_semesters)),
                'has_summer': has_summer,
                'curriculum_structure': curriculum_structure  # Include for detailed analysis
            }
            
            print(f"📊 Program duration: {total_years} years, semesters: {result['semesters']}")
            return result
            
        except Exception as e:
            print(f"❌ Error counting years/semesters: {e}")
            return {'total_years': 0, 'semesters': [], 'has_summer': False}

    def _get_all_programs_with_year_semester_data(self) -> dict:
        """Get all programs with their year/semester information"""
        try:
            # Get all curriculum documents
            all_curriculum_docs = self._get_all_curriculum_documents()
            
            programs_data = {}
            
            for doc in all_curriculum_docs:
                # Extract program acronym
                program_acronym = self._extract_program_acronym_from_curriculum(doc['content'])
                
                if program_acronym:
                    # Count years and semesters
                    year_semester_data = self._count_years_and_semesters_from_curriculum(doc['content'])
                    
                    # Get program details from JSON
                    program_details = self._get_program_details_from_json(program_acronym)
                    
                    programs_data[program_acronym] = {
                        'acronym': program_acronym,
                        'full_name': program_details.get('full_name', program_acronym),
                        'school': program_details.get('school', ''),
                        'cluster': program_details.get('cluster', ''),
                        'total_years': year_semester_data['total_years'],
                        'semesters': year_semester_data['semesters'],
                        'has_summer': year_semester_data['has_summer'],
                        'document_id': doc['id']
                    }
            
            print(f"📚 Analyzed {len(programs_data)} programs for year/semester data")
            return programs_data
            
        except Exception as e:
            print(f"❌ Error getting programs with year/semester data: {e}")
            return {}

    def _get_program_details_from_json(self, program_acronym: str) -> dict:
        """Get program details from JSON config by acronym"""
        try:
            config = self._load_normalization_config()
            program_abbrevs = config.get('program_abbreviations', {})
            
            # Search through all schools for the program
            for school_section, school_programs in program_abbrevs.items():
                if school_section.startswith('_'):  # Skip metadata
                    continue
                    
                for abbrev_key, abbrev_data in school_programs.items():
                    if isinstance(abbrev_data, dict) and abbrev_key.upper() == program_acronym.upper():
                        return {
                            'full_name': abbrev_data.get('full_name', program_acronym),
                            'school': abbrev_data.get('school', ''),
                            'cluster': abbrev_data.get('cluster', ''),
                            'description': abbrev_data.get('description', '')
                        }
            
            # If not found, return basic info
            return {
                'full_name': program_acronym,
                'school': '',
                'cluster': '',
                'description': ''
            }
            
        except Exception as e:
            print(f"❌ Error getting program details from JSON: {e}")
            return {'full_name': program_acronym, 'school': '', 'cluster': '', 'description': ''}

    def retrieve_programs_by_year_semester(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve programs based on year/semester requirements
        Handles queries like:
        - "what programs have 4 years"
        - "programs that are 3-4 years"
        - "programs that have no summer semester"
        """
        print(f"🎓 Year/semester-based program retrieval for: '{query}'")
        
        # Detect year/semester query details
        query_info = self._detect_year_semester_query(query)
        
        if not query_info['is_year_semester_query']:
            print("❌ Not a year/semester query")
            return []
        
        # Get all programs with their year/semester data
        all_programs_data = self._get_all_programs_with_year_semester_data()
        
        if not all_programs_data:
            print("❌ No program data available")
            return []
        
        # Apply cluster/school filtering if specified in query
        cluster_school_info = self._detect_cluster_or_school_in_query(query)
        if cluster_school_info['type'] in ['cluster', 'school']:
            print(f"🎯 Applying {cluster_school_info['type']} filter: {cluster_school_info['value']}")
            filtered_programs = {}
            for program_key, program_data in all_programs_data.items():
                if cluster_school_info['type'] == 'cluster':
                    if program_data['cluster'].lower() == cluster_school_info['value'].lower():
                        filtered_programs[program_key] = program_data
                elif cluster_school_info['type'] == 'school':
                    if program_data['school'].lower() == cluster_school_info['value'].lower():
                        filtered_programs[program_key] = program_data
            all_programs_data = filtered_programs
            print(f"📚 Filtered to {len(all_programs_data)} programs")
        
        # Filter programs based on year/semester requirements
        matching_programs = {}
        
        for program_key, program_data in all_programs_data.items():
            matches = False
            
            if query_info['query_type'] == 'exact_years':
                # Exact year match
                if program_data['total_years'] == query_info['exact_years']:
                    matches = True
                    print(f"✅ {program_key}: {program_data['total_years']} years (exact match)")
                
            elif query_info['query_type'] == 'year_range':
                # Year range match
                total_years = program_data['total_years']
                if query_info['min_years'] <= total_years <= query_info['max_years']:
                    matches = True
                    print(f"✅ {program_key}: {total_years} years (within range {query_info['min_years']}-{query_info['max_years']})")
                
            elif query_info['query_type'] == 'semester':
                # Semester requirement match
                semester_type = query_info['semester_type']
                has_semester = semester_type in program_data['semesters']
                
                if query_info['semester_requirement'] == 'include':
                    # Must have the semester
                    if has_semester:
                        matches = True
                        print(f"✅ {program_key}: has {semester_type} semester")
                elif query_info['semester_requirement'] == 'exclude':
                    # Must NOT have the semester
                    if not has_semester:
                        matches = True
                        print(f"✅ {program_key}: no {semester_type} semester")
            
            if matches:
                matching_programs[program_key] = program_data
        
        print(f"🎯 Found {len(matching_programs)} matching programs")
        
        # Convert to the expected format for the chatbot response
        result_data = {
            'matching_programs': matching_programs,
            'query_info': query_info,
            'cluster_school_filter': cluster_school_info,
            'total_programs': len(matching_programs)
        }
        
        # Return in the expected format
        return [{
            'id': 'year_semester_query_raw_data',
            'matching_programs': matching_programs,
            'query_info': query_info,
            'cluster_school_filter': cluster_school_info,
            'total_programs': len(matching_programs)
        }]

    def _format_year_semester_response(self, matching_programs: dict, query_info: dict) -> str:
        """Format response for year/semester queries"""
        try:
            if not matching_programs:
                return "I couldn't find any programs matching your year/semester criteria."
            
            query_type = query_info.get('query_type', 'unknown')
            total_programs = len(matching_programs)
            
            # Build response header based on query type
            if query_type == 'exact_years':
                years = query_info.get('exact_years', 'unknown')
                header = f"**Programs with {years} years duration:**\n\n"
            elif query_type == 'year_range':
                min_years = query_info.get('min_years', 'unknown')
                max_years = query_info.get('max_years', 'unknown')
                header = f"**Programs with {min_years}-{max_years} years duration:**\n\n"
            elif query_type == 'semester':
                semester_type = query_info.get('semester_type', 'unknown')
                semester_req = query_info.get('semester_requirement', 'include')
                if semester_req == 'exclude':
                    header = f"**Programs without {semester_type} semester:**\n\n"
                else:
                    header = f"**Programs with {semester_type} semester:**\n\n"
            else:
                header = "**Matching programs:**\n\n"
            
            # Group programs by school for better organization
            programs_by_school = {}
            for program_key, program_data in matching_programs.items():
                school = program_data.get('school', 'Unknown School')
                if school not in programs_by_school:
                    programs_by_school[school] = []
                programs_by_school[school].append(program_data)
            
            # Build response content
            response_parts = [header]
            
            for school, programs in programs_by_school.items():
                if len(programs_by_school) > 1:  # Only show school headers if multiple schools
                    response_parts.append(f"**{school}:**\n")
                
                for program_data in programs:
                    acronym = program_data.get('acronym', 'Unknown')
                    full_name = program_data.get('full_name', acronym)
                    cluster = program_data.get('cluster', '')
                    years = program_data.get('total_years', 'Unknown')
                    semesters = program_data.get('semesters', [])
                    
                    # Format program entry
                    program_line = f"• **{acronym}** - {full_name}"
                    if cluster:
                        program_line += f" ({cluster})"
                    program_line += f"\n  - Duration: {years} years"
                    if semesters:
                        semester_list = ', '.join([s.title() for s in semesters])
                        program_line += f"\n  - Semesters: {semester_list}"
                    program_line += "\n\n"
                    
                    response_parts.append(program_line)
            
            # Add summary
            response_parts.append(f"**Total:** {total_programs} program{'s' if total_programs != 1 else ''} found.\n\n")
            response_parts.append("*Need more details about any program? Just ask about the specific program name!*")
            
            return ''.join(response_parts)
            
        except Exception as e:
            print(f"❌ Error formatting year/semester response: {e}")
            return f"Found {len(matching_programs)} matching programs, but encountered an error formatting the response."

    def _format_duration_response(self, program_acronym: str, full_name: str, actual_years: int) -> str:
        """Format simple duration response for duration-only queries"""
        try:
            if actual_years <= 0:
                return f"I couldn't determine the duration for the {program_acronym} program. Please try asking about the curriculum for more details."
            
            # Build the response
            if full_name and full_name != program_acronym:
                response = f"The {full_name} ({program_acronym}) program at Ateneo de Davao University is a {actual_years}-year undergraduate degree program."
            else:
                response = f"The {program_acronym} program at Ateneo de Davao University is a {actual_years}-year undergraduate degree program."
            
            return response
            
        except Exception as e:
            print(f"❌ Error formatting duration response: {e}")
            return f"I encountered an error while retrieving the duration for {program_acronym}. Please try again."
    
    def _get_school_keywords(self):
        """Retrieve school keywords from the loaded configuration"""
        config = self._load_normalization_config()
        return config.get("school_keywords", {})
    
    def _get_cluster_keywords(self):
        """Retrieve cluster keywords from the loaded configuration"""
        config = self._load_normalization_config()
        return config.get("cluster_keywords", {})
    
    def _get_priority_for_abbrev(self, abbrev: str) -> str:
        """Get normalization priority for abbreviation"""
        abbreviations = self._get_program_abbreviations()
        abbrev_data = abbreviations.get(abbrev.lower(), {})
        return abbrev_data.get("priority", "safe")
    
    def _is_common_english_word(self, word: str) -> bool:
        """Check if word is a common English word that needs context checking"""
        config = self._load_normalization_config()
        common_words = config.get("common_english_words", [])
        return word.lower() in common_words
    
    def _resolve_abbrev_conflict(self, query: str, abbrev: str) -> str:
        """Resolve conflicts between abbreviations (e.g., AB IS vs BS IS)"""
        config = self._load_normalization_config()
        conflicts = config.get("conflict_resolution", {})
        
        if abbrev.lower() in conflicts:
            # For now, use simple heuristics - could be enhanced with ML
            query_lower = query.lower()
            
            # Check for computer/technology context for IS/DS conflicts
            tech_keywords = ["computer", "technology", "information", "data", "programming", "software", "system"]
            arts_keywords = ["islamic", "international", "development", "social", "studies", "culture"]
            
            if any(keyword in query_lower for keyword in tech_keywords):
                # Prefer BS version (Computer Studies)
                if abbrev.lower() == "is":
                    return "BS IS"
                elif abbrev.lower() == "ds":
                    return "BS DS"
            elif any(keyword in query_lower for keyword in arts_keywords):
                # Prefer AB version (Arts/Social Sciences)
                if abbrev.lower() == "is":
                    return "AB IS"
                elif abbrev.lower() == "ds":
                    return "AB DS"
        
        # Default: return the first match from abbreviations
        abbreviations = self._get_program_abbreviations()
        abbrev_data = abbreviations.get(abbrev.lower(), {})
        return abbrev_data.get("full_name", abbrev.upper())
    
    def _should_skip_normalization_for_subject_context(self, word: str, query: str) -> bool:
        """Check if word should skip normalization due to subject mapping context"""
        query_lower = query.lower()
        
        # Subject mapping patterns that should prevent program normalization
        subject_patterns = [
            f'{word} subjects',
            f'{word} courses', 
            f'about {word}',
            f'what about {word}',
            f'{word} topics',
            f'the {word} subjects',
            f'show me {word}',
            f'list {word}'
        ]
        
        # Check if any subject pattern matches
        for pattern in subject_patterns:
            if pattern in query_lower:
                print(f"🚫 Skipping normalization of '{word}' - detected subject context: '{pattern}'")
                return True
        
        return False

    def _normalize_program_acronyms(self, query: str) -> str:
        """Normalize program acronyms using configurable, priority-based system
        
        ENHANCED: Uses JSON configuration with 3-tier priority system:
        - safe: Always normalize (e.g., bscs -> BS CS)
        - context_aware: Normalize with program keywords (e.g., cs near 'program')
        - context_required: Only with strong context (e.g., 'is' only if 'BS IS program')
        
        ENHANCED: Skips normalization for subject mapping contexts (e.g., "math subjects")
        """
        import re
        
        query_lower = query.lower()
        
        # Load configuration
        context_patterns = self._get_context_patterns()
        abbreviations = self._get_program_abbreviations()
        
        # Detect context types using config patterns
        problematic_patterns = context_patterns.get("problematic", [])
        program_keywords = context_patterns.get("program_keywords", [])
        strong_program_patterns = context_patterns.get("strong_program", [])
        
        has_problematic_context = any(re.search(pattern, query_lower) for pattern in problematic_patterns)
        has_program_context = any(re.search(rf'\b{keyword}\b', query_lower) for keyword in program_keywords)
        
        # Build normalization mappings based on priority
        program_mappings = {}
        
        for abbrev, abbrev_data in abbreviations.items():
            if not isinstance(abbrev_data, dict):
                continue
                
            priority = abbrev_data.get("priority", "safe")
            full_name = abbrev_data.get("full_name", abbrev.upper())
            is_common_word = abbrev_data.get("is_common_word", False)
            
            # Check if this abbreviation should be skipped due to subject context
            if self._should_skip_normalization_for_subject_context(abbrev, query):
                continue  # Skip normalization for this abbreviation
            
            # Apply 3-tier logic
            should_normalize = False
            
            if priority == "safe":
                # Always normalize safe abbreviations
                should_normalize = True
            elif priority == "context_aware":
                # Normalize if program context is present
                should_normalize = has_program_context
            elif priority == "context_required":
                # Only normalize with strong explicit context
                has_strong_context = False
                
                # Check for strong program context patterns with this specific abbreviation
                for pattern_template in strong_program_patterns:
                    pattern = pattern_template.replace("{abbrev}", re.escape(abbrev))
                    if re.search(pattern, query_lower):
                        has_strong_context = True
                        break
                
                # For common English words, be extra conservative
                if is_common_word:
                    should_normalize = has_strong_context and not has_problematic_context
                else:
                    should_normalize = has_strong_context or (has_program_context and not has_problematic_context)
            
            # Add to mappings if should normalize
            if should_normalize:
                # Handle conflicts (e.g., AB IS vs BS IS)
                if abbrev_data.get("conflicts_with"):
                    resolved_name = self._resolve_abbrev_conflict(query, abbrev)
                    program_mappings[rf'\b{re.escape(abbrev)}\b'] = resolved_name
                else:
                    program_mappings[rf'\b{re.escape(abbrev)}\b'] = full_name
        
        # Apply normalizations (case insensitive)
        # Sort patterns by length (longest first) to avoid double replacements
        sorted_patterns = sorted(program_mappings.items(), key=lambda x: len(x[0]), reverse=True)
        
        # IMPORTANT: Apply all normalizations to the ORIGINAL query to avoid double normalization
        # Track positions that have been replaced to avoid overlapping replacements
        normalized_query = query
        changes_made = []
        replaced_positions = set()
        
        for pattern, replacement in sorted_patterns:
            # Find all matches in the ORIGINAL query
            matches = list(re.finditer(pattern, query, flags=re.IGNORECASE))
            
            for match in reversed(matches):  # Process from right to left to maintain positions
                start, end = match.span()
                
                # Check if this position has already been replaced
                if any(pos in replaced_positions for pos in range(start, end)):
                    continue
                
                # Apply the replacement
                normalized_query = normalized_query[:start] + replacement + normalized_query[end:]
                changes_made.append(f"{match.group()} -> {replacement}")
                
                # Mark these positions as replaced
                replaced_positions.update(range(start, end))
        
        # Log normalization decisions for debugging
        if changes_made:
            print(f"[NORM] Applied normalizations: {', '.join(changes_made)}")
            print(f"[NORM] '{query}' -> '{normalized_query}'")
        
        return normalized_query

    def _normalize_school_abbreviations(self, query: str) -> str:
        """Normalize school abbreviations in queries (e.g., SBG -> School of Business & Governance)"""
        import re
        
        # Load school abbreviations from config
        config = self._load_normalization_config()
        school_abbreviations = config.get("school_abbreviations", {})
        
        if not school_abbreviations:
            return query
        
        query_lower = query.lower()
        normalized_query = query
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_abbrevs = sorted(school_abbreviations.items(), key=lambda x: len(x[0]), reverse=True)
        
        for abbrev, full_name in sorted_abbrevs:
            if abbrev in query_lower:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(abbrev) + r'\b'
                if re.search(pattern, query_lower):
                    normalized_query = re.sub(pattern, full_name, normalized_query, flags=re.IGNORECASE)
                    print(f"[SCHOOL] Normalized: '{abbrev}' -> '{full_name}'")
        
        return normalized_query

    def _get_comprehensive_program_keywords(self) -> List[str]:
        """Get comprehensive list of program keywords including abbreviations"""
        return [
            # Business and Governance
            'bpm', 'bsa', 'bsma', 'bsbm', 'bsentrep', 'bsfin', 'bshrdm', 'bsmktg',
            'public management', 'accountancy', 'management accounting', 'business management', 
            'entrepreneurship', 'finance', 'human resource development management', 'marketing',
            
            # Arts and Sciences - Technology (with abbreviations)
            'bsit', 'bscs', 'bsis', 'bsds', 'bs it', 'bs cs', 'bs is', 'bs ds',
            'information technology', 'computer science', 'information systems', 'data science',
            # Abbreviations for Technology programs
            'it', 'cs', 'is', 'ds', 'compsci', 'infotech', 'datasci',
            
            # Arts and Sciences - Science (with abbreviations)
            'bsbio', 'bschem', 'bsmath', 'bsenvisci', 'bssocialwork', 'bs bio', 'bs chem', 'bs math', 'bs envisci', 'bs social work',
            'biology', 'chemistry', 'mathematics', 'environmental science', 'social work',
            # Abbreviations for Science programs
            'bio', 'chem', 'math', 'envisci', 'socialwork',
            
            # Arts and Sciences - Arts (with abbreviations)
            'abanthro', 'abanth', 'abc', 'abcomm', 'abds', 'abecon', 'abel', 'abis', 'abphilo', 'abpolsci', 'abpsych', 'absocio',
            'ab anthro', 'ab c', 'ab ds', 'ab econ', 'ab el', 'ab is', 'ab philo', 'ab polsci', 'ab psych', 'ab socio',
            'anthropology', 'communication', 'development studies', 'economics', 'english language', 
            'interdisciplinary studies', 'international studies', 'islamic studies', 'philosophy', 
            'political science', 'psychology', 'sociology',
            # Abbreviations for Arts programs
            'anthro', 'anth', 'comm', 'econ', 'el', 'philo', 'polsci', 'psych', 'socio',
            
            # Education (with abbreviations)
            'bece', 'beed', 'bsed', 'early childhood education', 'elementary education', 'secondary education',
            # Abbreviations for Education programs
            'ece', 'eed', 'sed',
            
            # Engineering and Architecture (with abbreviations)
            'bsae', 'bsarch', 'bsche', 'bsce', 'bscompeng', 'bscpe', 'bsee', 'bbselectronicseng', 'bsie', 'bsme', 'bsre',
            'bs ae', 'bs arch', 'bs che', 'bs ce', 'bs comp eng', 'bs ee', 'bs electronics eng', 'bs ie', 'bs me', 'bs re',
            'aerospace engineering', 'architecture', 'chemical engineering', 'civil engineering', 
            'computer engineering', 'electrical engineering', 'electronics engineering', 
            'industrial engineering', 'mechanical engineering', 'robotics engineering',
            # Abbreviations for Engineering programs
            'ae', 'arch', 'che', 'ce', 'compeng', 'cpe', 'ee', 'electronicseng', 'ie', 'me', 're',
            'aerospace', 'chemical', 'civil', 'computer', 'electrical', 'electronics', 'industrial', 'mechanical', 'robotics',
            
            # Nursing (with abbreviations)
            'bsn', 'nursing', 'nurse'
        ]

    def _enhance_query_with_conversation_context(self, query: str, topic_id: str) -> str:
        """Enhance query with conversation context for follow-up questions"""
        if not self.dialogue_history:
            return query
        
        query_lower = query.lower().strip()
        
        # Check if this is a follow-up question with pronouns or vague references
        follow_up_indicators = [
            'them', 'it', 'these', 'those', 'where', 'how', 'when', 'what about',
            'submit them', 'apply for it', 'get them', 'where can i', 'how do i',
            'where to', 'how to', 'when to', 'what are they', 'are they'
        ]
        
        is_follow_up = any(indicator in query_lower for indicator in follow_up_indicators)
        
        if not is_follow_up:
            return query
        
        # Get the last conversation exchange
        last_exchange = self.dialogue_history[-1] if self.dialogue_history else None
        if not last_exchange:
            return query
        
        last_query = last_exchange.get('query', '').lower()
        last_response = last_exchange.get('response', '').lower()
        
        # Extract key context from the last exchange based on topic
        context_keywords = []
        
        if topic_id == 'admissions_enrollment':
            # Look for visa, document, application context
            if 'visa' in last_query or 'visa' in last_response:
                context_keywords.extend(['visa', 'student visa', 'special study permit'])
            if 'document' in last_query or 'document' in last_response:
                context_keywords.extend(['documents', 'requirements'])
            if 'application' in last_query or 'application' in last_response:
                context_keywords.extend(['application', 'apply'])
            if 'international' in last_query:
                context_keywords.extend(['international student', 'foreign student'])
            if 'transfer' in last_query:
                context_keywords.extend(['transfer student', 'transferee'])
            if 'scholar' in last_query:
                context_keywords.extend(['scholarship', 'scholar'])
        
        elif topic_id == 'programs_courses':
            # Look for program, curriculum, course context
            if 'curriculum' in last_query or 'curriculum' in last_response:
                context_keywords.extend(['curriculum', 'courses'])
            
            # Check if the last query was about program lists (school/cluster queries)
            was_program_list_query = self._is_program_list_query(last_query)
            
            if was_program_list_query:
                # Preserve program list context for follow-up queries
                context_keywords.extend(['programs', 'list'])
                
                # Extract school context from last query
                school_abbreviations = self._get_school_abbreviations()
                for abbrev, school_name in school_abbreviations.items():
                    if abbrev.lower() in last_query or school_name.lower() in last_query:
                        context_keywords.extend(['school programs', 'programs under'])
                        break
                
                # Extract cluster context from last query
                cluster_keywords = ['cluster', 'humanities', 'sciences', 'computer', 'business', 'engineering']
                if any(keyword in last_query for keyword in cluster_keywords):
                    context_keywords.extend(['cluster programs', 'programs in'])
            
            elif 'program' in last_query or 'program' in last_response:
                context_keywords.extend(['program', 'degree'])
                
            # Extract specific program names from last query
            import re
            program_patterns = [
                r'\b(bs|ba|ab|bpm|bsa|bsma|bsbm|bsit|bscs|bsis|bsds)\b',
                r'\b(computer science|information technology|business|nursing|engineering)\b'
            ]
            for pattern in program_patterns:
                matches = re.findall(pattern, last_query, re.IGNORECASE)
                context_keywords.extend(matches)
        
        elif topic_id == 'fees':
            # Look for fee, payment, tuition context
            if 'fee' in last_query or 'tuition' in last_query:
                context_keywords.extend(['fees', 'tuition', 'payment'])
            if 'cost' in last_query or 'price' in last_query:
                context_keywords.extend(['cost', 'price'])
        
        # Enhance the query with relevant context
        if context_keywords:
            # Choose the most relevant context keywords (max 2)
            relevant_context = ' '.join(context_keywords[:2])
            enhanced_query = f"{relevant_context} {query}"
            return enhanced_query
        
        return query

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
        
        # First check if this is a legitimate administrative process query (NOT private)
        administrative_process_keywords = [
            'grade appeal', 'appeal grade', 'appeal my grade', 'grade grievance',
            'appeal process', 'how to appeal', 'grade complaint', 'grade petition',
            'grade reconsideration', 'grade review', 'grade correction',
            'appeal procedure', 'appeal form', 'appeal deadline',
            'grade inquiry', 'grade dispute', 'contest grade'
        ]
        
        # If asking about administrative processes, it's NOT a privacy query
        for process_keyword in administrative_process_keywords:
            if process_keyword in query_lower:
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
                'entrance exam', 'interview', 'assessment', 'acat', 'ateneo college admissions test'
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
                'much', 'how much', 'expensive', 'cheap', 'afford', 'budget',
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
        fee_keywords = ['fees', 'tuition', 'payment', 'cost', 'price', 'amount', 'billing',
                        'much', 'how much', 'expensive', 'cheap', 'afford', 'budget']
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
    
    def _get_school_programs(self, school_id: str) -> str:
        """Get program list for a specific school dynamically from JSON config"""
        # Map school IDs to school names
        school_id_to_name = {
            'school_arts_sciences': 'School of Arts & Sciences',
            'school_business_governance': 'School of Business & Governance',
            'school_education': 'School of Education',
            'school_engineering_architecture': 'School of Engineering & Architecture',
            'school_nursing': 'School of Nursing'
        }
        
        school_name = school_id_to_name.get(school_id)
        if not school_name:
            return "School information not found."
        
        # Get programs for this school from JSON config
        school_programs = self._get_programs_by_school_from_config(school_name)
        
        if not school_programs:
            return f"No programs found for {school_name}."
        
        # Format the output
        output = f"**{school_name}**\n\n"
        
        for cluster, programs in school_programs.items():
            if programs:  # Only show clusters that have programs
                output += f"● {cluster} (Cluster)\n"
                for i, program in enumerate(programs, 1):
                    output += f"{i}. {program['abbreviation']} – {program['description']}\n"
                output += "\n"
        
        return output.strip()
    
    def _parse_curriculum_by_years(self, curriculum_content: str) -> dict:
        """
        Parse curriculum content and organize by year levels and semesters.
        
        Returns:
            dict: {year: {semester: [courses]}} structure
        """
        import re
        
        print(f"📚 Parsing curriculum content ({len(curriculum_content)} characters)")
        
        # Combined year-semester patterns (most common format)
        combined_year_semester_patterns = [
            # With "Semester" word
            r'([IVX]+)\.\s*Year\s+(\d+)\s*[—\-–]\s*(First|Second|Summer)\s+Semester',
            r'(\d+)\.\s*Year\s+(\d+)\s*[—\-–]\s*(First|Second|Summer)\s+Semester',
            # Without "Semester" word (e.g., "III. Year 1 — Summer (6.0 CU)")
            r'([IVX]+)\.\s*Year\s+(\d+)\s*[—\-–]\s*(First|Second|Summer)\s*\(',
            r'(\d+)\.\s*Year\s+(\d+)\s*[—\-–]\s*(First|Second|Summer)\s*\('
        ]
        
        # Separate year patterns (fallback)
        year_patterns = [
            r'([IVX]+)\.\s*Year\s+(\d+)',
            r'(\d+)\.\s*Year\s+(\d+)',
            r'Year\s+(\d+)',
            r'(\d+)(?:st|nd|rd|th)\s+Year'
        ]
        
        # Separate semester patterns (fallback)
        semester_patterns = [
            r'(First|Second|Summer)\s+Semester',
            r'(\d+)(?:st|nd|rd|th)\s+Semester',
            r'Semester\s+(\d+)'
        ]
        
        # Course patterns - extract course codes and titles (improved patterns)
        course_patterns = [
            # Pattern 1: Simple numbered courses (most reliable) - supports mixed case codes
            r'(\d+)\.\s*([A-Za-z]{2,}\s+\d+[A-Za-z]*)\s*[-–—]\s*(.+)',
            # Pattern 2: Numbered courses with optional credits in parentheses - supports mixed case codes
            r'(\d+)\.\s*([A-Za-z]{2,}\s+\d+[A-Za-z]*)\s*[-–—]\s*([^(\n]+?)(?:\s*\(([^)]+)\))?',
            # Pattern 3: Course codes without numbers (fallback) - supports mixed case codes
            r'([A-Za-z]{2,}\s+\d+[A-Za-z]*)\s*[-–—]\s*(.+)'
        ]
        
        curriculum_structure = {}
        current_year = None
        current_semester = None
        
        lines = curriculum_content.split('\n')
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Skip page markers (various formats) - these interrupt curriculum flow
            page_marker_patterns = [
                r'^Page\s+\d+\s*[-–—]\s*[-–—]*$',  # "Page 2 - --" or "Page 2 -"
                r'^[-–—]+\s*Page\s+\d+\s*[-–—]+$',  # "--- Page 3 ---"
                r'^Page\s+\d+\s*[-–—]+$',  # "Page 2 ---"
                r'^[-–—]+\s*Page\s+\d+\s*[-–—]*$',  # "--- Page 3" or "--- Page 3 ---"
                r'^\d+\s*[-–—]+\s*Page\s*[-–—]+\s*\d+$',  # "2 -- Page -- 3"
                r'^Page\s+\d+$',  # Simple "Page 2"
            ]
            
            is_page_marker = False
            for pattern in page_marker_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    print(f"📄 Skipping page marker at line {line_num + 1}: '{line}'")
                    is_page_marker = True
                    break
            
            if is_page_marker:
                continue  # Skip page markers but maintain current year/semester context
            
            # Check for combined year-semester markers first (most common)
            combined_found = False
            for pattern in combined_year_semester_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    
                    # Extract year with enhanced error handling
                    try:
                        if len(groups) >= 2 and groups[1]:
                            year_str = groups[1]  # Second group is the year number
                            current_year = int(year_str)
                        else:
                            print(f"⚠️ Year group missing or empty in pattern match: {groups}")
                            current_year = 1
                    except (ValueError, IndexError) as e:
                        print(f"⚠️ Error extracting year from groups {groups}: {e}")
                        current_year = 1
                    
                    # Extract semester with enhanced error handling
                    try:
                        if len(groups) >= 3 and groups[2]:
                            semester_str = groups[2].lower()  # Third group is the semester
                        else:
                            print(f"⚠️ Semester group missing or empty in pattern match: {groups}")
                            continue  # Skip this match if no semester info
                    except (IndexError, AttributeError) as e:
                        print(f"⚠️ Error extracting semester from groups {groups}: {e}")
                        continue
                    
                    # Normalize semester names
                    if semester_str in ['first', '1st', '1']:
                        current_semester = 'first'
                    elif semester_str in ['second', '2nd', '2']:
                        current_semester = 'second'
                    elif semester_str in ['summer', '3rd', '3']:
                        current_semester = 'summer'
                    else:
                        current_semester = semester_str
                    
                    # Initialize structure
                    if current_year and current_year not in curriculum_structure:
                        curriculum_structure[current_year] = {}
                    
                    if current_semester and current_semester not in curriculum_structure[current_year]:
                        curriculum_structure[current_year][current_semester] = []
                    
                    print(f"📅 Found Year {current_year} - {current_semester.title()} Semester at line {line_num + 1}")
                    combined_found = True
                    break
            
            if combined_found:
                continue
            
            # Check for separate year markers (fallback)
            year_found = False
            for pattern in year_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    # Handle different year formats with enhanced error handling
                    try:
                        if len(match.groups()) >= 1 and match.group(-1):
                            year_str = match.group(-1)  # Last group is usually the year
                            # Convert Roman numerals to Arabic
                            if year_str in ['I', 'II', 'III', 'IV', 'V']:
                                roman_to_arabic = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5}
                                current_year = roman_to_arabic.get(year_str, 1)
                            else:
                                try:
                                    current_year = int(year_str)
                                except ValueError:
                                    print(f"⚠️ Could not convert year string '{year_str}' to integer")
                                    current_year = 1
                        else:
                            print(f"⚠️ Year group missing or empty in pattern match")
                            current_year = 1
                    except (IndexError, AttributeError) as e:
                        print(f"⚠️ Error extracting year from match: {e}")
                        current_year = 1
                    
                    if current_year and current_year not in curriculum_structure:
                        curriculum_structure[current_year] = {}
                    
                    print(f"📅 Found Year {current_year} at line {line_num + 1}")
                    year_found = True
                    break
            
            if year_found:
                continue
            
            # Check for separate semester markers (fallback)
            semester_found = False
            for pattern in semester_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match and current_year:
                    semester_str = match.group(1).lower()
                    
                    # Normalize semester names
                    if semester_str in ['first', '1st', '1']:
                        current_semester = 'first'
                    elif semester_str in ['second', '2nd', '2']:
                        current_semester = 'second'
                    elif semester_str in ['summer', '3rd', '3']:
                        current_semester = 'summer'
                    
                    if current_semester and current_semester not in curriculum_structure[current_year]:
                        curriculum_structure[current_year][current_semester] = []
                    
                    print(f"📝 Found {current_semester.title()} Semester for Year {current_year}")
                    semester_found = True
                    break
            
            if semester_found:
                continue
            
            # Check for course entries
            if current_year and current_semester:
                # Skip Pay Units/Credit Units lines (decimal numbering like "1.1.", "1.2.")
                if re.match(r'\d+\.\d+\.\s*(Pay Units|Credit Units|Prerequisite)', line, re.IGNORECASE):
                    continue
                
                # Double-check for page markers in course extraction (additional safety)
                is_page_marker_in_course = False
                for pattern in page_marker_patterns:
                    if re.match(pattern, line, re.IGNORECASE):
                        print(f"📄 Skipping page marker in course section at line {line_num + 1}: '{line}'")
                        is_page_marker_in_course = True
                        break
                
                if is_page_marker_in_course:
                    continue
                
                for i, pattern in enumerate(course_patterns):
                    match = re.search(pattern, line)
                    if match:
                        groups = match.groups()
                        
                        # Handle different pattern formats safely
                        try:
                            if i == 0:  # Pattern 1: simple numbered (most reliable)
                                if len(groups) >= 3:
                                    course_code = groups[1].strip()
                                    course_title = groups[2].strip()
                                    credits = ""
                                else:
                                    continue
                            elif i == 1:  # Pattern 2: numbered with optional credits
                                if len(groups) >= 3:
                                    course_code = groups[1].strip()
                                    course_title = groups[2].strip()
                                    credits = groups[3].strip() if len(groups) > 3 and groups[3] else ""
                                else:
                                    continue
                            elif i == 2:  # Pattern 3: fallback without numbers
                                if len(groups) >= 2:
                                    course_code = groups[0].strip()
                                    course_title = groups[1].strip()
                                    credits = ""
                                else:
                                    continue
                        except IndexError as e:
                            print(f"⚠️ Group index error for pattern {i+1} with {len(groups)} groups: {e}")
                            continue
                        
                        # Clean up course title
                        course_title = re.sub(r'\s+', ' ', course_title).strip()
                        
                        # Skip lines that are not actual courses (like credit unit lines)
                        if 'pay units' in course_title.lower() or 'credit units' in course_title.lower():
                            continue
                        
                        course_entry = {
                            'code': course_code,
                            'title': course_title,
                            'credits': credits
                        }
                        
                        curriculum_structure[current_year][current_semester].append(course_entry)
                        print(f"📝 Extracted course: {course_code} - {course_title}")
                        break
        
        # Log parsing results
        total_courses = sum(
            len(courses) 
            for year_data in curriculum_structure.values() 
            for courses in year_data.values()
        )
        
        print(f"📊 Parsed curriculum structure:")
        for year, semesters in curriculum_structure.items():
            for semester, courses in semesters.items():
                print(f"   Year {year} - {semester.title()}: {len(courses)} courses")
        
        print(f"📈 Total courses parsed: {total_courses}")
        
        return curriculum_structure
    
    def _extract_year_from_curriculum(self, curriculum_content: str, target_year: int) -> dict:
        """
        Extract specific year content from curriculum using regex patterns.
        
        Args:
            curriculum_content: Raw curriculum text
            target_year: Year level to extract (1, 2, 3, 4)
            
        Returns:
            dict: {semester: [courses]} for the target year
        """
        full_structure = self._parse_curriculum_by_years(curriculum_content)
        return full_structure.get(target_year, {})
    
    def _detect_curriculum_query_type(self, query: str) -> dict:
        """
        Detect the type of curriculum query and extract relevant information.
        
        Returns:
            dict: {
                'type': 'initial_curriculum|year_specific|year_followup',
                'year': int or None,
                'program': str or None,
                'confidence': float
            }
        """
        import re
        
        query_lower = query.lower()
        
        classification = {
            'type': 'initial_curriculum',
            'year': None,
            'program': None,
            'confidence': 0.0,
            'matched_patterns': []
        }
        
        # Follow-up patterns with year (check these FIRST to distinguish from year_specific)
        followup_year_patterns = [
            r'\bwhat\s+about\s+(?:the\s+|its\s+)?(\d+)(?:st|nd|rd|th)\s+year\b',
            r'\bwhat\s+about\s+(?:the\s+|its\s+)?year\s+(\d+)\b',
            r'\bwhat\s+about\s+(?:the\s+|its\s+)?(second|third|fourth|2nd|3rd|4th)\s+year\b',
        ]
        
        # Year-specific patterns (initial queries with explicit years)
        year_specific_patterns = [
            # Initial query patterns with explicit years (case insensitive program names)
            r'\bcurriculum\s+(?:of|for)\s+[A-Za-z\s]+\s+(?:for\s+)?(\d+)(?:st|nd|rd|th)\s+year\b',
            r'\bcurriculum\s+(?:of|for)\s+[A-Za-z\s]+\s+(?:for\s+)?(second|third|fourth|2nd|3rd|4th)\s+year\b',
            r'\bwhat\s+(?:is\s+)?(?:the\s+)?curriculum\s+(?:of|for)\s+[A-Za-z\s]+\s+(?:for\s+)?(\d+)(?:st|nd|rd|th)\s+year\b',
            r'\bwhat\s+(?:is\s+)?(?:the\s+)?curriculum\s+(?:of|for)\s+[A-Za-z\s]+\s+(?:for\s+)?(second|third|fourth|2nd|3rd|4th)\s+year\b',
            r'\bshow\s+(?:me\s+)?(?:the\s+)?curriculum\s+(?:of|for)\s+[A-Za-z\s]+\s+(?:for\s+)?(\d+)(?:st|nd|rd|th)\s+year\b',
            r'\bshow\s+(?:me\s+)?(?:the\s+)?curriculum\s+(?:of|for)\s+[A-Za-z\s]+\s+(?:for\s+)?(second|third|fourth|2nd|3rd|4th)\s+year\b',
            
            # General year patterns
            r'\b(\d+)(?:st|nd|rd|th)\s+year\s+(?:curriculum|subjects?|courses?)\b',
            r'\byear\s+(\d+)\s+(?:curriculum|subjects?|courses?)\b'
        ]
        
        # Follow-up patterns (without year)
        followup_patterns = [
            r'\bcan\s+you\s+proceed\s+with\s+(?:the\s+)?remaining\b',
            r'\bwhat\s+(?:are\s+)?(?:the\s+)?(?:next|remaining)\b',
            r'\bcontinue\s+with\s+(?:the\s+)?(?:curriculum|subjects?|courses?)\b',
            r'\bshow\s+(?:me\s+)?(?:the\s+)?(?:next|remaining)\s+(?:year|subjects?|courses?)\b',
            r'\bwhat\s+(?:comes\s+)?next\b',
            r'\bproceed\b',
            r'\bwhat\'?s\s+next\b',
            r'\bshow\s+(?:me\s+)?(?:the\s+)?remaining\s+years?\b'
        ]
        
        # Initial curriculum patterns
        initial_patterns = [
            r'\bcurriculum\s+(?:of|for)\s+([A-Z\s]+)\b',
            r'\bwhat\s+(?:is\s+)?(?:the\s+)?curriculum\s+(?:of|for)\s+([A-Z\s]+)\b',
            r'\bshow\s+(?:me\s+)?(?:the\s+)?curriculum\s+(?:of|for)\s+([A-Z\s]+)\b',
            r'\bwhat\s+(?:are\s+the\s+)?(?:subjects?|courses?)\s+(?:in|for|of)\s+([A-Z\s]+)\b'
        ]
        
        # Check for follow-up queries with year FIRST (to distinguish from year_specific)
        for pattern in followup_year_patterns:
            match = re.search(pattern, query_lower)
            if match:
                classification['type'] = 'year_followup'
                classification['confidence'] = 0.9
                classification['matched_patterns'].append(f"Follow-up with year: {pattern}")
                
                # Extract year number
                year_str = match.group(1)
                if year_str in ['second', '2nd']:
                    classification['year'] = 2
                elif year_str in ['third', '3rd']:
                    classification['year'] = 3
                elif year_str in ['fourth', '4th']:
                    classification['year'] = 4
                else:
                    try:
                        classification['year'] = int(year_str)
                    except ValueError:
                        classification['year'] = 1
                
                print(f"🔄 Detected follow-up query with year: Year {classification['year']}")
                return classification
        
        # Check for year-specific queries (initial queries with explicit years)
        for pattern in year_specific_patterns:
            match = re.search(pattern, query_lower)
            if match:
                classification['type'] = 'year_specific'
                classification['confidence'] = 0.9
                classification['matched_patterns'].append(f"Year-specific: {pattern}")
                
                # Extract year number
                year_str = match.group(1)
                if year_str in ['second', '2nd']:
                    classification['year'] = 2
                elif year_str in ['third', '3rd']:
                    classification['year'] = 3
                elif year_str in ['fourth', '4th']:
                    classification['year'] = 4
                else:
                    try:
                        classification['year'] = int(year_str)
                    except ValueError:
                        classification['year'] = 1
                
                print(f"📅 Detected year-specific query: Year {classification['year']}")
                return classification
        
        # Check for follow-up queries (without year)
        for pattern in followup_patterns:
            if re.search(pattern, query_lower):
                classification['type'] = 'year_followup'
                classification['confidence'] = 0.85
                classification['matched_patterns'].append(f"Follow-up: {pattern}")
                print(f"🔄 Detected follow-up query")
                return classification
        
        # Check for initial curriculum queries with program extraction
        for pattern in initial_patterns:
            match = re.search(pattern, query_lower)
            if match:
                classification['type'] = 'initial_curriculum'
                classification['confidence'] = 0.8
                classification['matched_patterns'].append(f"Initial: {pattern}")
                
                # Extract program name
                program_str = match.group(1).strip()
                classification['program'] = program_str
                
                print(f"📚 Detected initial curriculum query for: {program_str}")
                return classification
        
        # Default to initial curriculum
        classification['confidence'] = 0.7
        print(f"📋 Default classification: initial curriculum")
        
        return classification
    
    def _is_curriculum_followup_query(self, query: str) -> bool:
        """
        Detect if this is a curriculum follow-up query with year terms that might
        be confused with 'secondary education' in semantic similarity.
        """
        import re
        
        query_lower = query.lower()
        
        # Year-related terms that could be confused semantically
        year_terms = ['2nd', '3rd', '4th', 'second', 'third', 'fourth']
        
        # Follow-up patterns
        followup_patterns = [
            r'\bwhat\s+about\s+(?:the\s+|its\s+)?(?:2nd|3rd|4th|second|third|fourth)\s+year\b',
            r'\bwhat\s+about\s+(?:the\s+|its\s+)?year\s+(?:2|3|4)\b',
            r'\bcan\s+you\s+proceed\s+with\s+(?:the\s+)?remaining\b',
            r'\bshow\s+(?:me\s+)?(?:the\s+)?(?:next|remaining)\s+(?:year|subjects?|courses?)\b'
        ]
        
        # Check if query contains year terms AND follow-up patterns
        has_year_term = any(term in query_lower for term in year_terms)
        has_followup_pattern = any(re.search(pattern, query_lower) for pattern in followup_patterns)
        
        return has_year_term and has_followup_pattern
    





    def _process_topic_query(self, query: str, topic_id: str):
        """Process a query within a specific topic context with conversation history awareness"""
        try:
            # Enhanced program extraction with conversation history awareness
            program_info = self._extract_program_info_with_history(query)
            
            # Handle failed pronoun resolution
            if program_info.get('context_source') == 'pronoun_resolution_failed':
                return "I notice you're using a pronoun like 'it' or 'that', but I don't have enough context to understand which program you're referring to. Could you please specify the program name? For example: 'What is the curriculum for BS IT?' or 'What is the curriculum for BS CS?'", []
            
            # Normalize program acronyms first (e.g., 'bsa' -> 'BS A')
            normalized_query = self._normalize_program_acronyms(query)
            
            # First check for general conversation context enhancement (for follow-up questions)
            conversation_enhanced_query = self._enhance_query_with_conversation_context(normalized_query, topic_id)
            
            # Initialize enhanced query (will be modified based on intent)
            enhanced_query = conversation_enhanced_query
            
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
            
            # Use intent analysis combined with topic filtering for better accuracy
            intent = self.analyze_query_intent(enhanced_query)
            print(f"🎯 Intent analysis suggests document type: {intent.get('document_type', 'none')}")
            
            # For fees topic, use simple processing like the old system
            if topic_id == 'fees':
                print("💰 Processing fees query with simple approach")
                # Extract fee info using the simple method (already exists)
                fee_info = self._extract_fee_info(enhanced_query)
                print(f"💰 Extracted fee info: {fee_info}")
                
                # Use the normalized program for document retrieval if available
                if fee_info.get('program_name'):
                    self.set_session_state(current_program=fee_info['program_name'])
                    print(f"💰 Updated session state with program: {fee_info['program_name']}")
            
            # Calculate dynamic top_k based on intent and topic
            dynamic_top_k = 3  # Simple default
            print(f"🎯 Using top_k={dynamic_top_k} for retrieval")
            
            # Use specialized document retrieval
            relevant_docs = self.retrieve_documents_by_topic_specialized(enhanced_query, topic_id, top_k=dynamic_top_k)
            
            if not relevant_docs:
                return "I apologize, but I couldn't find relevant information for your query. Please try rephrasing your question or contact the admissions office for assistance.", []
            
            # Build context from retrieved docs
            doc_context = "\n\n".join([
                f"Source: {doc.get('id','')}\n{doc['content']}"
                for doc in relevant_docs[:dynamic_top_k]
            ])
            
            # Build history context for conversation continuity
            history_context = ""
            if self.dialogue_history:
                # Calculate available token budget for history
                base_prompt_estimate = f"System instructions + Context: {doc_context} + Query: {enhanced_query}"
                base_tokens = len(base_prompt_estimate.split())
                available_for_history = 3500 - base_tokens  # Conservative token limit
                
                # Use smart history building
                history_context = self.build_smart_history_context(query, available_for_history)
                print(f"📜 Built history context: {len(history_context)} chars (~{len(history_context.split())} tokens)")
            else:
                history_context = ""
            
            # Build prompt with topic context and specialized instructions
            topic_info = get_topic_info(topic_id)
            topic_label = topic_info.get('label', topic_id) if topic_info else topic_id
            
            # Build topic-specific instructions
            topic_specific_instructions = self._get_topic_specific_instructions(topic_id)
            
            prompt = f"""<|system|>
You are an ADDU (Ateneo de Davao University) Admissions Assistant. You provide accurate, helpful information based strictly on the provided context documents.

CRITICAL URL RULE: 
- NEVER create, invent, fabricate, or hallucinate URLs
- If no URLs are present in the source documents, do NOT mention links, URLs, or any reference to external resources
- Use topic-specific link handling rules (see topic instructions below)

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
- NEVER use tables, charts, or markdown table format
- Convert any tabular information to bullet points or numbered lists
- For age/program/visa or any other combinations, use clear bullet point format instead of tables
- Follow topic-specific link handling rules (see topic instructions)

CONTEXT MATCHING:
- Only use information that directly matches the user's specific query
- If context contains multiple student types/programs but user asked about one specific type, filter accordingly
- Prioritize exact matches over general information
</|system|>

<|context|>
{doc_context}
</|context|>

{history_context}

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
            print(f"❌ Error in _process_topic_query: {e}")
            return "I apologize, but I encountered an error while processing your query. Please try again or contact the admissions office for assistance.", []

    def _calculate_dynamic_top_k(self, topic_id: str, intent: str = None) -> int:
        """Calculate dynamic top_k based on topic and intent"""
        try:
            # Simple dynamic top_k calculation
            if topic_id == 'programs_courses':
                if intent in ['overview', 'mixed']:
                    return 5  # Need more documents for comprehensive overview
                elif intent in ['specific', 'curriculum']:
                    return 3  # Focused queries need fewer, more relevant docs
                else:
                    return 3  # Default for programs
            elif topic_id == 'fees':
                return 3  # Standard for fees queries
            elif topic_id == 'admissions_enrollment':
                return 4  # Admissions often needs multiple document types
            else:
                return 3  # Default
                
        except Exception as e:
            print(f"⚠️ Error calculating dynamic top_k: {e}, falling back to default top_k=3")
            return 3

    def retrieve_documents_by_topic_specialized(self, query: str, topic_id: str, top_k: int = 2) -> List[Dict]:
        """
        Simplified dispatcher that uses hybrid retrieval only.
        Specialized handlers have been removed to rely on hybrid retrieval + LLM formatting.
        """
        print(f"🎯 Using hybrid-only retrieval for topic: {topic_id}")
        
        # Always use hybrid retrieval (no specialized handler routing)
        if hasattr(self, 'use_hybrid_topic_retrieval') and self.use_hybrid_topic_retrieval:
            print("✅ Using unified hybrid retrieval (JSON-enhanced TF-IDF + Word2Vec)")
            return self.retrieve_documents_by_topic_hybrid(query, topic_id, top_k)
        else:
            print("📊 Using legacy hybrid retrieval")
            return self.retrieve_documents_by_topic_keywords(query, topic_id, top_k)

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
    


    
    def _get_next_year_level(self, curriculum_query_type: dict) -> int:
        """
        Determine the next year level to show based on conversation history and query type.
        
        Args:
            curriculum_query_type: Result from _detect_curriculum_query_type
            
        Returns:
            int: Year level to display (1, 2, 3, 4)
        """
        # Get curriculum state from session
        curriculum_state = self.session_state.get('curriculum_state', {})
        last_displayed_year = curriculum_state.get('last_displayed_year')
        available_years = curriculum_state.get('available_years', [1, 2, 3, 4])
        
        # If specific year requested, return that year
        if curriculum_query_type.get('year'):
            return curriculum_query_type['year']
        
        # For initial curriculum queries, always start with year 1
        if curriculum_query_type['type'] == 'initial_curriculum':
            return 1
        
        # For follow-up queries, determine next year
        if curriculum_query_type['type'] == 'year_followup':
            if last_displayed_year is None:
                return 1  # Start with first year if no previous display
            
            # Find next available year
            next_year = last_displayed_year + 1
            if next_year in available_years and next_year <= 4:
                return next_year
            else:
                # If no next year available, cycle back or show message
                return 1  # Could also return None to indicate no more years
        
        # Default to year 1
        return 1
    
    
    def _update_curriculum_session_state(self, program: str, displayed_year: int, parsed_curriculum: dict = None):
        """
        Update session state with curriculum information.
        
        Args:
            program: Program acronym/name
            displayed_year: Year level that was displayed
            parsed_curriculum: Optional parsed curriculum structure
        """
        if 'curriculum_state' not in self.session_state:
            self.session_state['curriculum_state'] = {}
        
        curriculum_state = self.session_state['curriculum_state']
        curriculum_state['current_program'] = program
        curriculum_state['last_displayed_year'] = displayed_year
        
        if parsed_curriculum:
            curriculum_state['parsed_curriculum'] = parsed_curriculum
            curriculum_state['available_years'] = list(parsed_curriculum.keys())
        
        print(f"📚 Updated curriculum session state: program={program}, last_year={displayed_year}")
    
    
    def _update_subject_mapping_session_state(self, program: str = None, subject_type: str = None):
        """Update session state for subject mapping queries with program and subject type tracking"""
        if 'subject_mapping_state' not in self.session_state:
            self.session_state['subject_mapping_state'] = {}
        
        subject_state = self.session_state['subject_mapping_state']
        
        # Update program context
        if program:
            subject_state['current_program'] = program
            self.session_state['current_program'] = program  # Cross-query continuity
        
        # Update subject type context
        if subject_type:
            subject_state['last_subject_type'] = subject_type
        
        # Track context history for ambiguity detection
        if 'context_history' not in subject_state:
            subject_state['context_history'] = []
        
        if program or subject_type:
            context_entry = {
                'program': program or subject_state.get('current_program'),
                'subject_type': subject_type or subject_state.get('last_subject_type'),
                'timestamp': time.time()
            }
            subject_state['context_history'].append(context_entry)
            
            # Keep only last 5 context entries
            subject_state['context_history'] = subject_state['context_history'][-5:]
        
        print(f"🔍 Updated subject mapping session state: program={program}, subject_type={subject_type}")
    
    def _build_general_subject_response(self, matching_programs: dict, subject_info: dict, is_negative: bool) -> str:
        """
        Build response for general positive/negative queries (e.g., "what programs offer math")
        Returns simple program list with full names from JSON config
        """
        print(f"🏗️ Building general subject response for {len(matching_programs)} programs")
        
        # Extract program acronyms and get full names from JSON config
        program_full_names = []
        for program_acronym in matching_programs.keys():
            full_name = self._get_program_full_name_from_config(program_acronym)
            if full_name:
                program_full_names.append(full_name)
                print(f"   ✅ {program_acronym} → {full_name}")
            else:
                print(f"   ❌ {program_acronym} → No full name found in config")
        
        # Sort programs alphabetically
        program_full_names.sort()
        
        # Build response
        subject = subject_info['subject']
        if is_negative:
            response = f"Here are the programs that DO NOT offer {subject}:\n\n"
        else:
            response = f"Here are the programs that offer {subject}:\n\n"
        
        # Add each program on its own line
        for program_name in program_full_names:
            response += f"• **{program_name}**\n"
        
        response += "\nFor more detailed information, please contact the admissions office."
        
        print(f"📝 Generated response with {len(program_full_names)} programs")
        return response
    
    def _build_detailed_subject_response(self, matching_programs: dict, subject_info: dict, is_negative: bool, cluster_filter: str = None) -> str:
        """
        Build response for program-specific and count queries (e.g., "how many math subjects in BS CS")
        Returns detailed subject information with course codes
        """
        print(f"🔍 Building detailed subject response for query type: {subject_info.get('query_type', 'unknown')}")
        
        # Use the existing _format_subject_mapping_response method for detailed queries
        formatted_response = self._format_subject_mapping_response(
            matching_programs, subject_info, is_negative, cluster_filter
        )
        
        if formatted_response and len(formatted_response) > 0:
            response = formatted_response[0]['content']
            print(f"📋 Generated detailed response: {len(response)} characters")
            return response
        else:
            subject = subject_info['subject']
            program = subject_info.get('program', 'the specified program')
            response = f"I couldn't find any {subject} courses in {program}. This subject may not be offered in this program."
            print(f"❌ No detailed results found, returning fallback response")
            return response
    
    def _process_topic_query(self, query: str, topic_id: str):
        """Process a query within a specific topic context with conversation history awareness"""
        try:
            # Initialize programs_query_intent at the start (will be set later if needed)
            programs_query_intent = None
            
            # Normalize program acronyms once at the start (before topic-specific logic)
            normalized_query = self._normalize_program_acronyms(query)
            
            # Check if this is an overview query FIRST (before any program extraction)
            if topic_id == 'programs_courses':
                query_context = self._detect_cluster_or_school_in_query(normalized_query)
                overview_keywords = ['available', 'list', 'show', 'all programs', 'what are the programs', 'what programs']
                is_overview_query = (
                    query_context['type'] == 'general' and 
                    any(word in normalized_query.lower() for word in overview_keywords)
                )
                
                if is_overview_query:
                    print(f"📋 Overview query detected in _process_topic_query - skipping program extraction")
                    # Set empty program_info for overview queries
                    program_info = {
                        'program_name': None,
                        'degree_level': None,
                        'year_level': None,
                        'course_code': None,
                        'context_source': None
                    }
                else:
            # Enhanced program extraction with conversation history awareness
                    program_info = self._extract_program_info_with_history(query)
            else:
                # For other topics, extract program info normally
                program_info = self._extract_program_info_with_history(query)
            
            # Handle failed pronoun resolution
            if program_info.get('context_source') == 'pronoun_resolution_failed':
                return "I notice you're using a pronoun like 'it' or 'that', but I don't have enough context to understand which program you're referring to. Could you please specify the program name? For example: 'What is the curriculum for BS IT?' or 'What is the curriculum for BS CS?'", []
            
            # normalized_query already set at line 7218, reuse it
            
            # First check for general conversation context enhancement (for follow-up questions)
            conversation_enhanced_query = self._enhance_query_with_conversation_context(normalized_query, topic_id)
            
            # Initialize enhanced query (will be modified based on intent)
            enhanced_query = conversation_enhanced_query
            
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
            
            # For programs topic, check for special query types first, then validate program availability
            if topic_id == 'programs_courses':
                # FIRST: Check for year/semester queries BEFORE intent classification
                year_semester_info = self._detect_year_semester_query(query)
                if year_semester_info['is_year_semester_query']:
                    print("🎓 Detected year/semester query - using flexible retrieval (bypassing strict intent classification)")
                    # Skip intent classification and go directly to retrieval
                    # This will be handled by retrieve_documents_by_topic_specialized()
                    intent = "year_semester"  # Special marker to bypass subject mapping logic
                else:
                    # SECOND: Check for duration-only queries BEFORE intent classification
                    duration_info = self._detect_duration_query(query)
                    if duration_info['is_duration_query']:
                        print("⏱️ Detected duration query - using simple duration response (bypassing curriculum logic)")
                        intent = "duration"  # Special marker to bypass curriculum logic
                    else:
                        # Only classify intent if NOT a year/semester or duration query
                        intent = self._classify_programs_query_intent(query)
                
                # INTENT-AWARE PROGRAM CONTEXT EXTRACTION: Only extract program context for relevant intents
                if intent == "year_semester":
                    # For year/semester queries, clear program context to ensure flexible retrieval
                    program_info = {
                        'program_name': None,
                        'degree_level': None,
                        'year_level': None,
                        'course_code': None,
                        'context_source': None
                    }
                    print(f"🎓 Year/semester query - cleared program context for flexible retrieval")
                elif intent == "duration":
                    # For duration queries, clear program context to ensure simple response
                    program_info = {
                        'program_name': None,
                        'degree_level': None,
                        'year_level': None,
                        'course_code': None,
                        'context_source': None
                    }
                    print(f"⏱️ Duration query - cleared program context for simple response")
                elif intent == "subject_mapping":
                    # Check if this is actually an overview query misclassified as subject_mapping
                    normalized_query_check = self._normalize_program_acronyms(query)
                    query_context_check = self._detect_cluster_or_school_in_query(normalized_query_check)
                    overview_keywords = ['available', 'list', 'show', 'all programs', 'what are the programs', 'what programs']
                    is_actually_overview = (
                        query_context_check['type'] == 'general' and 
                        any(word in normalized_query_check.lower() for word in overview_keywords)
                    )
                    
                    if is_actually_overview:
                        # This was misclassified as subject_mapping, but it's actually an overview query
                        print(f"⚠️ Query misclassified as subject_mapping but is actually overview - correcting")
                        program_info = {
                            'program_name': None,
                            'degree_level': None,
                            'year_level': None,
                            'course_code': None,
                            'context_source': None
                        }
                        intent = "overview"  # Correct the intent
                    else:
                        # For subject mapping queries, ONLY use program context if explicitly mentioned in current query
                        current_program_info = self._extract_program_info(query)
                        if current_program_info.get('program_name'):
                            program_info = current_program_info
                            print(f"📝 Subject mapping query explicitly mentions program: {program_info['program_name']}")
                        else:
                            # Clear any program context from history for subject mapping queries
                            program_info = {
                                'program_name': None,
                                'degree_level': None,
                                'year_level': None,
                                'course_code': None,
                                'context_source': None
                            }
                            print(f"🚫 Subject mapping query - cleared program context from history")
                
                # INTENT-AWARE CONTEXT APPLICATION: Apply program context based on query intent
                if program_info.get('program_name') and program_info.get('context_source'):
                    # For curriculum queries, apply program context (for follow-ups like "what about 2nd year")
                    if intent == "curriculum":
                        preprocessed_query = self._preprocess_pronoun_query(query, program_info['program_name'])
                        # Re-normalize after pronoun preprocessing
                        normalized_query = self._normalize_program_acronyms(preprocessed_query)
                        conversation_enhanced_query = self._enhance_query_with_conversation_context(normalized_query, topic_id)
                        
                        # Check if the current query already contains a program name
                        current_program_info = self._extract_program_info(preprocessed_query)
                        if not current_program_info.get('program_name'):
                            # Check if the program from history is already mentioned in the current query
                            program_name_lower = program_info['program_name'].lower()
                            if program_name_lower not in preprocessed_query.lower():
                                # Add program context for curriculum queries
                                if conversation_enhanced_query != normalized_query:
                                    enhanced_query = f"{program_info['program_name']} {conversation_enhanced_query}"
                                else:
                                    enhanced_query = f"{program_info['program_name']} {normalized_query}"
                                print(f"🎯 Enhanced curriculum query with program context from {program_info['context_source']}: '{query}' → '{enhanced_query}'")
                            else:
                                enhanced_query = conversation_enhanced_query
                                print(f"📝 Program from history already mentioned in curriculum query, using conversation-enhanced query: '{enhanced_query}'")
                        else:
                            enhanced_query = conversation_enhanced_query
                            print(f"📝 Curriculum query already contains program info, using conversation-enhanced query: '{enhanced_query}'")
                    
                    # For subject mapping queries, DO NOT apply program context unless explicitly mentioned
                    elif intent == "subject_mapping":
                        # Check if the current query explicitly mentions a program
                        current_program_info = self._extract_program_info(query)
                        if current_program_info.get('program_name'):
                            # Program explicitly mentioned, use it
                            enhanced_query = conversation_enhanced_query
                            print(f"📝 Subject mapping query explicitly mentions program, using conversation-enhanced query: '{enhanced_query}'")
                        else:
                            # No program mentioned, don't apply program context from history
                            enhanced_query = conversation_enhanced_query
                            print(f"🚫 Subject mapping query without explicit program - not applying program context from history")
                    
                    # For other intents (overview, specific, mixed), apply program context normally
                    else:
                        preprocessed_query = self._preprocess_pronoun_query(query, program_info['program_name'])
                        # Re-normalize after pronoun preprocessing
                        normalized_query = self._normalize_program_acronyms(preprocessed_query)
                        conversation_enhanced_query = self._enhance_query_with_conversation_context(normalized_query, topic_id)
                        
                        # Check if the current query already contains a program name
                        current_program_info = self._extract_program_info(preprocessed_query)
                        if not current_program_info.get('program_name'):
                            # Check if the program from history is already mentioned in the current query
                            program_name_lower = program_info['program_name'].lower()
                            if program_name_lower not in preprocessed_query.lower():
                                # Add program context for other query types
                                if conversation_enhanced_query != normalized_query:
                                    enhanced_query = f"{program_info['program_name']} {conversation_enhanced_query}"
                                else:
                                    enhanced_query = f"{program_info['program_name']} {normalized_query}"
                                print(f"🎯 Enhanced {intent} query with program context from {program_info['context_source']}: '{query}' → '{enhanced_query}'")
                            else:
                                enhanced_query = conversation_enhanced_query
                                print(f"📝 Program from history already mentioned in {intent} query, using conversation-enhanced query: '{enhanced_query}'")
                        else:
                            enhanced_query = conversation_enhanced_query
                            print(f"📝 {intent} query already contains program info, using conversation-enhanced query: '{enhanced_query}'")
                else:
                    print(f"📝 No program context available for {intent} query, using conversation-enhanced query: '{enhanced_query}'")
                
                # REMOVED: Year/semester specialized routing - now handled by hybrid retrieval + LLM
                
                # PRIORITY -0.5: Handle duration queries (simple program duration response)
                if intent == "duration":
                    print("⏱️ Processing duration query - providing simple duration response")
                    try:
                        # Extract program from the duration query
                        program_name = duration_info.get('program')
                        if not program_name:
                            return "I couldn't identify the program you're asking about. Please specify a program name.", []
                        
                        # Normalize program name using existing method
                        normalized_program = self._normalize_program_acronyms(program_name)
                        
                        # Retrieve curriculum document for the program
                        curriculum_docs = self.retrieve_programs_documents(normalized_program, top_k=1)
                        
                        if not curriculum_docs:
                            return f"I couldn't find information for the {program_name} program. Please check the program name and try again.", []
                        
                        # Calculate actual duration from curriculum
                        curriculum_content = curriculum_docs[0]['content']
                        year_semester_data = self._count_years_and_semesters_from_curriculum(curriculum_content)
                        actual_years = year_semester_data.get('total_years', 0)
                        
                        # Get program details from JSON config
                        program_acronym = self._extract_program_acronym_from_curriculum(curriculum_content)
                        if not program_acronym:
                            program_acronym = normalized_program.upper()
                        
                        full_name = self._get_program_full_name_from_config(program_acronym)
                        
                        # Format simple duration response
                        response = self._format_duration_response(program_acronym, full_name, actual_years)
                        
                        return response, curriculum_docs
                        
                    except Exception as e:
                        print(f"❌ Duration query processing failed: {e}")
                        return "I encountered an error while retrieving the program duration. Please try again.", []
                
                # PRIORITY 0: Check for curriculum intent (e.g., "BS CS curriculum", "what about 2nd year")
                if intent == "curriculum":
                    print("📚 Processing curriculum query with context awareness")
                    try:
                        # Detect curriculum query type and determine target year
                        curriculum_query_type = self._detect_curriculum_query_type(query)
                        target_year = self._get_next_year_level(curriculum_query_type)
                        
                        print(f"📅 Target year: {target_year}")
                        print(f"📊 Query type: {curriculum_query_type['type']}")
                        
                        # For follow-up queries, enhance with program from session/conversation
                        if curriculum_query_type['type'] in ['year_specific', 'year_followup']:
                            program_info_from_query = self._extract_program_info_with_history(enhanced_query)
                            session_program = program_info_from_query.get('program_name')
                            
                            if not session_program:
                                session_program = self.session_state.get('current_program')
                            
                            if session_program:
                                # Enhance query with program context for curriculum
                                if session_program.lower() not in enhanced_query.lower():
                                    enhanced_query = f"{session_program} {enhanced_query}"
                                    print(f"📚 Enhanced curriculum query with program context: '{query}' → '{enhanced_query}'")
                                
                                # Add year context to query if not already present
                                if f"year {target_year}" not in enhanced_query.lower() and str(target_year) not in enhanced_query:
                                    enhanced_query = f"{enhanced_query} year {target_year}"
                                    print(f"📚 Enhanced curriculum query with year context: year {target_year}")
                                
                                    # Update session state
                                    self._update_curriculum_session_state(session_program, target_year)
                            else:
                                if curriculum_query_type['type'] == 'year_followup':
                                    return "I need to know which program you're asking about. Please specify a program name.", []
                        # For initial curriculum queries, enhance with program if extracted
                        elif curriculum_query_type['type'] == 'initial_curriculum':
                            program_info = self._extract_program_info_with_history(enhanced_query)
                            if program_info.get('program_name'):
                                program_name = program_info['program_name']
                                # Add year context (default to year 1 if not specified)
                                if f"year {target_year}" not in enhanced_query.lower():
                                    enhanced_query = f"{enhanced_query} year {target_year}"
                                    print(f"📚 Enhanced curriculum query with year context: year {target_year}")
                                
                                # Update session state
                                self._update_curriculum_session_state(program_name, target_year)
                                print(f"📚 Updated curriculum session state for initial query: {program_name}")
                                
                    except Exception as e:
                        print(f"❌ Curriculum context processing failed: {e}")
                        # Continue with normal processing if curriculum context fails

                # PRIORITY 2.5: Simplified subject mapping processing (removed subject type context awareness)
                if intent == 'subject_mapping':
                    print("🔍 Processing subject mapping query")
                    try:
                        # Get session context for pronoun preprocessing (basic program tracking only)
                        session_program = self.session_state.get('current_program')
                        
                        # Preprocess pronouns if we have program context
                        if session_program:
                            preprocessed_query = self._preprocess_pronoun_query(query, session_program)
                            if preprocessed_query != query:
                                enhanced_query = preprocessed_query
                                print(f"🔄 Preprocessed pronouns in subject mapping query: '{query}' → '{preprocessed_query}'")
                        
                        # Extract program from current query
                        program_info_from_query = self._extract_program_info_with_history(enhanced_query)
                        current_program = program_info_from_query.get('program_name')
                        
                        # Determine effective program (use current or session)
                        effective_program = current_program or session_program
                        
                        # Enhance query with program context if needed
                        if effective_program and effective_program.lower() not in enhanced_query.lower():
                            enhanced_query = f"{effective_program} {enhanced_query}"
                            print(f"🔍 Enhanced with program context: '{query}' → '{enhanced_query}'")
                        
                    except Exception as e:
                        print(f"❌ Subject mapping processing failed: {e}")
                        # Continue with normal processing
                
                # PRIORITY 2: Check program availability for specific program queries (not overview queries)
                availability_indicators = ['is there', 'do you have', 'available', 'offer', 'does addu have']
                
                if intent != 'overview' and intent != 'subject_mapping' and any(indicator in query.lower() for indicator in availability_indicators):
                    print(f"🔍 Detected specific program availability query: '{query}'")
                    
                    # First check our program mappings for quick validation
                    availability_result = self._parse_program_availability_configurable(query)
                    
                    if availability_result['exists'] == False:
                        # Program definitely doesn't exist - provide clear response
                        response_text = "Based on our official program list, that program is not currently offered at Ateneo de Davao University. For the most up-to-date list of available programs, please contact our admissions office."
                        return response_text, []
                    
                    elif availability_result['exists'] == True:
                        # Check if this is an ambiguous query with multiple matches
                        if availability_result.get('is_ambiguous', False):
                            # Handle ambiguous query - provide multiple options
                            print(f"🔀 Ambiguous query detected, providing multiple program options")
                            
                            response_parts = ["I found multiple programs that match your query. Here are the available options:\n"]
                            
                            for group_key, programs in availability_result['program_groups'].items():
                                response_parts.append(f"\n**{group_key}:**")
                                for program in programs:
                                    response_parts.append(f"• **{program['program_name']}** - {program['description']}")
                            
                            response_parts.append(f"\nCould you please specify which program you're interested in? You can ask about any of these {availability_result['total_matches']} programs by name.")
                            
                            return "\n".join(response_parts), []
                        
                        else:
                            # Single program match - enhance query with program context
                            program_context = f"Program: {availability_result['details']}"
                            if availability_result['school']:
                                program_context += f"\nSchool: {availability_result['school']}"
                            if availability_result['cluster']:
                                program_context += f"\nCluster: {availability_result['cluster']}"
                            
                            enhanced_query = f"{enhanced_query}\n\nProgram Context: {program_context}"
                            print(f"✅ Program exists, enhanced query with context")
                    elif intent == 'overview':
                        print(f"🔍 Detected overview query, skipping program availability check: '{query}'")
                    
                    # For general program list queries, try to retrieve the program list document
                    elif self._is_program_list_query(query):
                        print(f"🔍 Detected program list query, retrieving official program list")
                        program_list_doc = self._retrieve_program_list_document()
                        
                        if program_list_doc['found']:
                            # Format and return the program list response
                            formatted_response = self._format_program_list_response(program_list_doc['content'], query)
                            return formatted_response, [{'id': 'program_list_official', 'content': program_list_doc['content'], 'metadata': program_list_doc['metadata']}]
                        else:
                            print("⚠️ Could not retrieve program list document")
            
            # Save programs query intent before it gets overwritten by analyze_query_intent
            # Note: For programs_courses, intent is set earlier as a STRING ('overview', 'curriculum', etc.)
            # But analyze_query_intent() at line 7661 overwrites it with a DICT
            if topic_id == 'programs_courses' and isinstance(intent, str):
                programs_query_intent = intent
                print(f"💾 Saved programs query intent: '{programs_query_intent}'")
            
            # Use intent analysis combined with topic filtering for better accuracy
            intent = self.analyze_query_intent(enhanced_query)
            print(f"🎯 Intent analysis suggests document type: {intent.get('document_type', 'none')}")
            
            
            # Calculate dynamic top_k based on intent and topic
            # Use programs_query_intent if available (for programs_courses), otherwise use default
            if topic_id == 'programs_courses' and programs_query_intent:
                dynamic_top_k = self._calculate_dynamic_top_k(topic_id, programs_query_intent)
            else:
                # For non-programs topics or if programs intent not available, use default
                dynamic_top_k = 3
                
            print(f"🎯 Using top_k={dynamic_top_k} for retrieval")
            
            # SPECIALIZED ROUTING: Use subject extraction logic for subject mapping queries
            if topic_id == 'programs_courses' and programs_query_intent == 'subject_mapping':
                print("🔍 Using specialized subject extraction for subject mapping query")
                relevant_docs = self.retrieve_programs_by_subject(enhanced_query, top_k=dynamic_top_k)
            else:
                # Use general hybrid retrieval for other intents (curriculum, overview, etc.)
                print("🎯 Using hybrid retrieval for non-subject-mapping query")
                relevant_docs = self.retrieve_documents_by_topic_specialized(enhanced_query, topic_id, top_k=dynamic_top_k)
            
            if not relevant_docs:
                topic_info = get_topic_info(topic_id)
                topic_label = topic_info.get('label', topic_id) if topic_info else topic_id
                response_text = f"I don't have specific information about that in the {topic_label} topic. Could you try rephrasing your question?"
                
                return response_text, []
            
            # Detect if this is a simple factual question that needs a short answer (moved here for scope)
            simple_question_patterns = [
                r'how many\s+\w+\s+are\s+there',  # "how many summers are there"
                r'how many\s+\w+\s+does\s+\w+\s+have',  # "how many years does bscs have"
                r'what\s+is\s+the\s+\w+\s+of',  # "what is the duration of"
                r'how\s+long\s+is',  # "how long is the program"
                r'when\s+is\s+the\s+\w+',  # "when is the deadline"
                r'where\s+is\s+the\s+\w+',  # "where is the office"
                r'what\s+time\s+\w+',  # "what time does it open"
                r'how\s+much\s+\w+',  # "how much does it cost"
                r'is\s+there\s+\w+',     # "is there summer"
                r'does\s+\w+\s+have\s+', # "does bs cs have summer"
                r'are\s+there\s+\w+',    # "are there summer classes"
                r'is\s+\w+\s+required',  # "is summer required"
                r'is\s+\w+\s+mandatory', # "is summer mandatory"
                r'do\s+i\s+need\s+to\s+', # "do I need to take"
            ]
            
            # Detect if this is a yes/no question that needs answer-first format
            yes_no_question_patterns = [
                r'do\s+i\s+still\s+need\s+to',  # "do I still need to take an entrance exam"
                r'do\s+i\s+need\s+to\s+',  # "do I need to take"
                r'do\s+you\s+have\s+',  # "do you have"
                r'does\s+\w+\s+have\s+',  # "does the program have"
                r'does\s+\w+\s+require\s+',  # "does it require"
                r'is\s+there\s+\w+',  # "is there an entrance exam"
                r'are\s+there\s+\w+',  # "are there requirements"
                r'is\s+\w+\s+required',  # "is summer required"
                r'is\s+\w+\s+mandatory',  # "is it mandatory"
                r'is\s+\w+\s+available',  # "is it available"
                r'is\s+\w+\s+offered',  # "is it offered"
                r'can\s+i\s+',  # "can I apply"
                r'will\s+i\s+',  # "will I need"
                r'should\s+i\s+',  # "should I take"
                r'must\s+i\s+',  # "must I submit"
                r'do\s+transferees\s+',  # "do transferees need"
                r'do\s+new\s+students\s+',  # "do new students need"
                r'do\s+international\s+students\s+',  # "do international students need"
                r'does\s+addu\s+',  # "does addu require"
                r'are\s+entrance\s+exams\s+',  # "are entrance exams required"
                r'is\s+an\s+entrance\s+exam\s+',  # "is an entrance exam required"
            ]
            
            import re
            is_simple_question = any(re.search(pattern, enhanced_query.lower()) for pattern in simple_question_patterns)
            is_yes_no_question = any(re.search(pattern, enhanced_query.lower()) for pattern in yes_no_question_patterns)
            
            # For simple questions and yes/no questions, use only the top document to avoid overwhelming context
            # For subject mapping queries, use more documents to capture all programs with the subject
            # For complex questions, use up to 3 documents for comprehensive answers
            if programs_query_intent == 'subject_mapping':
                docs_to_use = min(len(relevant_docs), 15)  # Use up to 15 documents for comprehensive subject mapping
                print(f"🎯 Subject mapping query detected - using {docs_to_use} documents for comprehensive coverage")
            elif is_simple_question or is_yes_no_question:
                docs_to_use = 1
            else:
                docs_to_use = 3
            
            # Build context from retrieved docs (use full content to preserve URLs)
            if relevant_docs and relevant_docs[0].get('id') == 'subject_mapping_organized_data':
                # Handle organized subject mapping data
                doc_data = relevant_docs[0]
                matching_programs = doc_data.get('matching_programs', {})
                
                # Convert organized data to readable context for LLM
                doc_context_parts = []
                for program_name, program_data in matching_programs.items():
                    doc_context_parts.append(f"Program: {program_name}")
                    
                    if 'subjects_by_year_semester' in program_data:
                        # Use organized structure
                        year_semester_data = program_data['subjects_by_year_semester']
                        for year in sorted(year_semester_data.keys()):
                            if year != 'Unknown':
                                doc_context_parts.append(f"\nYear {year}:")
                                for semester, subjects in year_semester_data[year].items():
                                    if semester and subjects:
                                        if semester == 'unknown':
                                            semester_name = "Unspecified Semester"
                                        else:
                                            semester_name = semester.title() if isinstance(semester, str) else str(semester)
                                            semester_name += " Semester"
                                        
                                        doc_context_parts.append(f"  {semester_name}:")
                                        for subject in subjects:
                                            doc_context_parts.append(f"    {subject['full']}")
                    else:
                        # Fallback to flat structure
                        all_subjects = program_data.get('all_subjects', [])
                        for subject in all_subjects:
                            year = subject.get('year', 'Unknown')
                            semester = subject.get('semester', 'unknown')
                            doc_context_parts.append(f"    {subject['full']} (Year {year}, {semester})")
                
                doc_context = "\n".join(doc_context_parts)
                print(f"📊 Built organized context for {len(matching_programs)} programs")
            else:
                # Original logic for regular documents
                doc_context = "\n\n".join([
                    f"Source: {doc.get('id','')}\n{doc['content']}"  # Use full content - no truncation
                    for doc in relevant_docs[:docs_to_use]
                ])
            
            if is_yes_no_question:
                print(f"🎯 Yes/No question detected - using only top document: {relevant_docs[0].get('filename', 'Unknown') if relevant_docs else 'None'}")
            elif is_simple_question:
                print(f"🎯 Simple question detected - using only top document: {relevant_docs[0].get('filename', 'Unknown') if relevant_docs else 'None'}")
            
            # Build prompt with topic context and specialized instructions
            topic_info = get_topic_info(topic_id)
            topic_label = topic_info.get('label', topic_id) if topic_info else topic_id
            
            # Build topic-specific instructions
            topic_specific_instructions = self._get_topic_specific_instructions(topic_id)
            
            # Add intent-specific instructions for subject mapping
            intent_specific_instruction = ""
            if programs_query_intent == 'subject_mapping':
                # Extract subject type from query for dynamic title
                subject_type = self._extract_subject_type_from_query(enhanced_query)
                
                # Format subject type for display (capitalize first letter)
                subject_type_display_map = {
                    'programming': 'Programming',
                    'mathematics': 'Math',
                    'science': 'Science',
                    'english': 'English',
                    'business': 'Business',
                    'engineering': 'Engineering',
                    'general': 'Subject'
                }
                subject_type_display = subject_type_display_map.get(subject_type, subject_type.capitalize())
                
                intent_specific_instruction = f"""
🎯 SUBJECT MAPPING INTENT DETECTED - CRITICAL FORMATTING OVERRIDE 🎯

*** IGNORE ALL CURRICULUM FORMATTING RULES FOR THIS QUERY ***
*** DO NOT USE "Year 1" TITLE FORMAT ***
*** DO NOT DEFAULT TO YEAR 1 CURRICULUM ***

MANDATORY SUBJECT MAPPING BEHAVIOR:
- This query is asking about programs that have a specific subject OR subjects within a specific program
- FOCUS ONLY on programs/subjects that are directly relevant to the query
- EXCLUDE general program listings, school/cluster breakdowns, or unrelated program information
- Use the SUBJECT MAPPING QUERIES section in the topic instructions above
- Trust the semantic matching - the retrieved documents already contain relevant programs/subjects

CRITICAL FORMATTING OVERRIDE FOR PROGRAM-SPECIFIC QUERIES:
- TITLE FORMAT: Use "**{subject_type_display} Subjects in [Program Name]**" (NO YEAR SPECIFICATION, WITH BOLD FORMATTING)
- YEAR FORMATTING: Use bold formatting for years (e.g., "**Year 1:**", "**Year 2:**")
- SEMESTER FORMATTING: Use bold formatting for semesters (e.g., "**First Semester:**", "**Second Semester:**", "**Summer Semester:**")
- CONTENT SCOPE: Show subjects from ALL YEARS (Year 1, 2, 3, 4) - NEVER limit to Year 1
- ORGANIZATION: The data is pre-organized by year and semester - USE ALL AVAILABLE YEARS
- DATA STRUCTURE: subjects_by_year_semester contains Year 1, 2, 3, 4 data - SHOW ALL OF IT
- IGNORE: Any "default to Year 1" instructions from curriculum section
- IGNORE: Any year-based display rules from curriculum section
- MANDATORY: If you see organized year/semester data, display ALL years, not just Year 1

RESPONSE LENGTH FOR SUBJECT MAPPING:
- For GENERAL queries: Provide comprehensive list of ALL relevant programs organized by School and Cluster
- For PROGRAM-SPECIFIC queries: Show detailed subjects across ALL YEARS with course codes and descriptions
- COMPREHENSIVE COVERAGE: Include all relevant subjects from the entire program duration
- Do NOT limit to top 3 programs - include ALL programs found in the documents
- Use bullet point format with proper School/Cluster organization

*** REPEAT: DO NOT USE CURRICULUM YEAR 1 DEFAULT BEHAVIOR ***
*** REPEAT: SHOW ALL YEARS FOR SUBJECT MAPPING QUERIES ***
"""
            
            # Add response length instruction based on question complexity
            response_length_instruction = ""
            if is_yes_no_question:
                response_length_instruction = """
YES/NO QUESTION FORMAT: This is a yes/no question. You MUST follow this exact format:
1. Start with a clear YES or NO answer
2. Then provide a brief explanation (1-2 sentences)
3. Do NOT provide extensive background information first
4. Follow topic-specific link handling rules (see topic instructions)

Example format:
"NO, you do not need to take an entrance exam as a transferee. [Brief explanation here]"

CRITICAL: The YES/NO answer must come FIRST, before any explanation."""
            elif is_simple_question:
                response_length_instruction = """
RESPONSE LENGTH: This is a simple factual question. Give a SHORT, DIRECT answer (1-2 sentences maximum). Do NOT provide extensive background information, full curriculum details, or comprehensive explanations unless specifically asked."""
            
            prompt = f"""<|system|>
You are an ADDU (Ateneo de Davao University) Admissions Assistant. You provide accurate, helpful information based strictly on the provided context documents.

CRITICAL URL RULE: 
- NEVER create, invent, fabricate, or hallucinate URLs
- If no URLs are present in the source documents, do NOT mention links, URLs, or any reference to external resources
- Use topic-specific link handling rules (see topic instructions below)

{topic_specific_instructions}

{intent_specific_instruction}

GENERAL RESPONSE RULES:
- Be direct and concise
- Use simple formatting
- No introductory phrases like "Based on the provided documentation"
- No closing phrases like "I hope this helps"
- Start directly with the answer
- Use numbered lists for steps
- Use bullet points for items
- Bold important terms only when necessary
- NEVER use tables, charts, or markdown table format
- Convert any tabular information to bullet points or numbered lists
- For age/program/visa or any other combinations, use clear bullet point format instead of tables
- Follow topic-specific link handling rules (see topic instructions)

CONTEXT MATCHING:
- Only use information that directly matches the user's specific query
- If context contains multiple student types/programs but user asked about one specific type, filter accordingly
- Prioritize exact matches over general information

{response_length_instruction}
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
            # Use higher token limit for subject mapping queries to ensure comprehensive coverage
            max_tokens_limit = 5000 if programs_query_intent == 'subject_mapping' else 3000
            
            if programs_query_intent == 'subject_mapping':
                print(f"🎯 Subject mapping query - using {max_tokens_limit} token limit and {docs_to_use} documents")
            
            full_response = ""
            for chunk in stream_response_together(prompt, max_tokens=max_tokens_limit):
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

    # REMOVED: process_query_stream - using guided chatbot only
    def _removed_process_query_stream(self, query: str, correct_spelling: bool = True, max_tokens: int = 5000,
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

    def _parse_program_availability_configurable(self, query: str) -> Dict:
        """
        Parse program availability using the configurable normalization system
        Uses intelligent matching with scoring to find the best program match
        """
        config = self._load_normalization_config()
        abbreviations = self._get_program_abbreviations()
        
        import re
        # Remove punctuation for better program matching
        query_clean = re.sub(r'[^\w\s]', '', query)
        query_lower = query_clean.lower().strip()
        
        # Find all potential matches with scores
        matches = []
        
        for abbrev, abbrev_data in abbreviations.items():
            if not isinstance(abbrev_data, dict):
                continue
                
            full_name = abbrev_data.get("full_name", "")
            description = abbrev_data.get("description", "")
            
            # Calculate match score for this program
            match_score = self._calculate_program_match_score(query_lower, abbrev, full_name, description)
            
            if match_score > 0:
                # Apply priority filtering (same as _extract_program_info)
                priority = abbrev_data.get("priority", "safe")
                is_common_word = abbrev_data.get("is_common_word", False)
                should_include = False
                
                if priority == "safe":
                    should_include = True
                elif priority == "context_aware":
                    # Check for program context
                    context_patterns = self._get_context_patterns()
                    program_keywords = context_patterns.get("program_keywords", [])
                    has_program_context = any(re.search(rf'\b{keyword}\b', query_lower) for keyword in program_keywords)
                    should_include = has_program_context
                elif priority == "context_required":
                    # Only include with strong context (for common English words)
                    context_patterns = self._get_context_patterns()
                    strong_program_patterns = context_patterns.get("strong_program", [])
                    has_strong_context = False
                    
                    for pattern_template in strong_program_patterns:
                        pattern = pattern_template.replace("{abbrev}", re.escape(abbrev))
                        if re.search(pattern, query_lower):
                            has_strong_context = True
                            break
                    
                    # For common English words, be extra conservative
                    if is_common_word:
                        problematic_patterns = context_patterns.get("problematic", [])
                        has_problematic_context = any(re.search(pattern, query_lower) for pattern in problematic_patterns)
                        
                        # SMART DETECTION: Check if the abbreviation appears as uppercase in original query
                        # This handles cases like "is there IS" where "IS" is clearly a program reference
                        has_uppercase_abbrev = abbrev.upper() in query
                        
                        # If the abbreviation appears in uppercase, it's likely a program reference
                        if has_uppercase_abbrev:
                            should_include = True  # Override filtering for uppercase program references
                        else:
                            should_include = has_strong_context and not has_problematic_context
                    else:
                        should_include = has_strong_context
                
                if should_include:
                    matches.append({
                        'score': match_score,
                        'abbrev': abbrev,
                        'full_name': full_name,
                        'description': description,
                        'abbrev_data': abbrev_data
                    })
        
        # Check if both "is" and "ds" are present for tie-breaking
        has_is = any(m['abbrev'] == "is" for m in matches)
        has_ds = any(m['abbrev'] == "ds" for m in matches)
        
        # Sort matches by score (highest first), then by abbreviation length (longer = more specific)
        # For ties, prioritize based on query context
        if matches:
            def tie_breaker(match):
                score = match['score']
                length = len(match['abbrev'])
                abbrev = match['abbrev']
                
                # If scores and lengths are equal, prioritize based on query context
                if score >= 0.9 and length == 2:  # Both are short abbreviations
                    if abbrev == "ds" and has_is:
                        return (score, length, 1)  # Give ds a boost
                    elif abbrev == "is" and has_ds:
                        return (score, length, 0)  # Give is lower priority
                
                return (score, length, 0)
            
            sorted_matches = sorted(matches, key=tie_breaker, reverse=True)
            best_match = sorted_matches[0]
            
            # Check for ambiguous queries (multiple high-scoring matches)
            if len(sorted_matches) > 1:
                # Get matches with similar high scores (within 0.1 of the best)
                high_scoring_matches = [
                    m for m in sorted_matches 
                    if m['score'] >= best_match['score'] - 0.1 and m['score'] > 0.3
                ]
                
                # If we have multiple high-scoring matches, check if they're different programs
                if len(high_scoring_matches) > 1:
                    unique_programs = set(m['full_name'] for m in high_scoring_matches)
                    
                    if len(unique_programs) > 1:
                        # Handle ambiguous query - return multiple options
                        print(f"[AMBIGUOUS] Found {len(unique_programs)} matching programs for query: {query}")
                        
                        # Group matches by school/cluster for better organization
                        program_groups = {}
                        for match in high_scoring_matches:
                            school = self._extract_school_from_config(match['abbrev_data'])
                            cluster = self._extract_cluster_from_config(match['abbrev_data'])
                            key = f"{school} - {cluster}" if school and cluster else "Unknown"
                            
                            if key not in program_groups:
                                program_groups[key] = []
                            
                            program_groups[key].append({
                                'program_name': match['full_name'],
                                'description': match['description'],
                                'school': school,
                                'cluster': cluster
                            })
                        
                        return {
                            'exists': True,
                            'is_ambiguous': True,
                            'program_groups': program_groups,
                            'total_matches': len(unique_programs)
                        }
            
            # Only return if the match score is significant enough (> 0.3)
            if best_match['score'] > 0.3:
                print(f"[MATCH] Found program '{best_match['full_name']}' with score {best_match['score']:.3f}")
                return {
                    'exists': True,
                    'is_ambiguous': False,
                    'program_name': best_match['full_name'],
                    'details': best_match['description'],
                    'school': self._extract_school_from_config(best_match['abbrev_data']),
                    'cluster': self._extract_cluster_from_config(best_match['abbrev_data'])
                }
        
        return {
            'exists': False,
            'is_ambiguous': False,
            'program_name': None,
            'details': None,
            'school': None,
            'cluster': None
        }
    
    def _calculate_program_match_score(self, query_lower: str, abbrev: str, full_name: str, description: str) -> float:
        """
        Calculate match score for a program based on query
        Returns score between 0.0 and 1.0, where higher is better
        """
        import re
        
        score = 0.0
        
        # Extract program name from description (e.g., "Data Science" from "Bachelor of Science in Data Science")
        program_name = self._extract_program_name_from_description(description)
        
        # 1. Exact program name match (highest priority) - Score: 1.0
        if program_name and program_name.lower() in query_lower:
            # Check if it's a word boundary match (not partial)
            if re.search(r'\b' + re.escape(program_name.lower()) + r'\b', query_lower):
                score = 1.0
                return score
        
        # 2. Exact abbreviation match - Score: 0.9 (with length bonus for specificity)
        if abbrev.lower() in query_lower:
            if re.search(r'\b' + re.escape(abbrev.lower()) + r'\b', query_lower):
                # Give bonus for longer, more specific abbreviations
                length_bonus = min(0.1, len(abbrev) * 0.01)  # Up to 0.1 bonus for longer abbrevs
                score = 0.9 + length_bonus
                return score
        
        # 2.5. Handle space-separated abbreviations (e.g., "ab anthro" matches "abanthro")
        # Check if abbreviation without spaces matches query with spaces removed
        import re
        abbrev_no_spaces = abbrev.replace(' ', '').lower()
        # Remove punctuation from query for better matching
        query_no_spaces = re.sub(r'[^\w\s]', '', query_lower).replace(' ', '')
        
        if abbrev_no_spaces in query_no_spaces:
            # Additional check: ensure the abbreviation parts appear in sequence
            abbrev_parts = abbrev.lower().split()
            if len(abbrev_parts) > 1:  # Only for multi-word abbreviations
                query_parts = query_lower.split()
                
                # Check if all abbreviation parts appear in sequence in query
                for i in range(len(query_parts) - len(abbrev_parts) + 1):
                    if query_parts[i:i + len(abbrev_parts)] == abbrev_parts:
                        # Give bonus for longer, more specific abbreviations
                        length_bonus = min(0.1, len(abbrev) * 0.01)
                        score = 0.9 + length_bonus
                        return score
        else:
            # Single word abbreviation - check if it can be formed from query parts
            # For "abanthro", check if "ab" and "anthro" appear in sequence
            query_parts = query_lower.split()
            
            # Try to find the abbreviation by combining consecutive query parts
            for i in range(len(query_parts)):
                for j in range(i + 1, len(query_parts) + 1):
                    combined = ''.join(query_parts[i:j])
                    if combined == abbrev_no_spaces:
                        length_bonus = min(0.1, len(abbrev) * 0.01)
                        score = 0.9 + length_bonus
                        return score
        
        # 3. Exact full name match (e.g., "BS CS") - Score: 0.8
        # CRITICAL FIX: Check base abbreviation directly instead of full name
        # Extract the base abbreviation (e.g., "AB IS" from "AB IS - AMERICAN STUDIES" or "BS ENTREP" from "BS ENTREP-A")
        if ' - ' in full_name:
            base_abbrev = full_name.split(' - ')[0].strip()
        elif '-' in full_name and not full_name.startswith('-'):
            # Handle cases like "BS ENTREP-A" -> "BS ENTREP"
            base_abbrev = full_name.split('-')[0].strip()
        else:
            base_abbrev = full_name
        
        if base_abbrev.lower() in query_lower:
            if re.search(r'\b' + re.escape(base_abbrev.lower()) + r'\b', query_lower):
                # Check if the query contains this base abbreviation exactly
                query_parts = query_lower.split()
                base_parts = base_abbrev.lower().split()
                
                # Look for exact match of the base abbreviation in the query
                for i in range(len(query_parts) - len(base_parts) + 1):
                    query_segment = ' '.join(query_parts[i:i + len(base_parts)])
                    if query_segment == base_abbrev.lower():
                        score = 0.8
                        return score
        
        # 4. Partial program name match - Score: 0.4-0.6
        # IMPORTANT: Use word boundaries to avoid substring matches (e.g., "science" matching "political science")
        if program_name:
            program_words = program_name.lower().split()
            # Count matches WITH word boundaries (avoid substring matches)
            matched_words = sum(1 for word in program_words 
                               if re.search(r'\b' + re.escape(word) + r'\b', query_lower))
            if matched_words > 0:
                score = 0.4 + (matched_words / len(program_words)) * 0.2
                # For multi-word programs, require at least 2 words or all words to match
                if len(program_words) > 1 and matched_words < len(program_words):
                    # Penalize partial matches for multi-word programs
                    score *= 0.7  # Reduce score if not all words match
        
        # 5. Description keyword match - Score: 0.1-0.3 (lowest priority)
        # IMPORTANT: Use word boundaries here too to avoid false matches
        description_words = [word.lower() for word in description.split() if len(word) > 4]
        matched_desc_words = sum(1 for word in description_words 
                                if re.search(r'\b' + re.escape(word) + r'\b', query_lower))
        if matched_desc_words > 0 and score < 0.3:
            score = max(score, 0.1 + (matched_desc_words / len(description_words)) * 0.2)
        
        return score
    
    def _extract_program_name_from_description(self, description: str) -> str:
        """
        Extract the actual program name from description
        E.g., "Bachelor of Science in Data Science" -> "Data Science"
        """
        import re
        
        # Common patterns to extract program names
        patterns = [
            r'Bachelor of (?:Science|Arts) in (.+?)(?:\s*\(|$)',
            r'Bachelor of (.+?)(?:\s*\(|$)',
            r'BS (.+?)(?:\s*\(|$)',
            r'BA (.+?)(?:\s*\(|$)',
            r'AB (.+?)(?:\s*\(|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                program_name = match.group(1).strip()
                # Clean up common suffixes
                program_name = re.sub(r'\s*Major in\s*', ' ', program_name, flags=re.IGNORECASE)
                return program_name
        
        return ""

    def _extract_school_from_config(self, abbrev_data: dict) -> str:
        """Extract school name from program config (explicit or inferred)"""
        # First check if school is explicitly defined in config
        if "school" in abbrev_data:
            return abbrev_data["school"]
        
        # Fallback to description-based inference
        description = abbrev_data.get("description", "").lower()
        
        if any(term in description for term in ["arts", "sciences", "english", "communication", "biology", "chemistry", "computer", "psychology", "anthropology", "economics", "sociology"]):
            return "School of Arts & Sciences"
        elif any(term in description for term in ["business", "accountancy", "management", "entrepreneurship", "finance", "marketing", "public"]):
            return "School of Business & Governance"
        elif any(term in description for term in ["education", "elementary", "secondary", "childhood"]):
            return "School of Education"
        elif any(term in description for term in ["engineering", "architecture"]):
            return "School of Engineering & Architecture"
        elif "nursing" in description:
            return "School of Nursing"
        else:
            return "Unknown School"

    def _extract_cluster_from_config(self, abbrev_data: dict) -> str:
        """Extract cluster name from program config (explicit or inferred)"""
        # First check if cluster is explicitly defined in config
        if "cluster" in abbrev_data:
            return abbrev_data["cluster"]
        
        # Fallback to description-based inference
        description = abbrev_data.get("description", "").lower()
        
        if any(term in description for term in ["english", "communication", "interdisciplinary", "philosophy"]):
            return "Humanities & Letters"
        elif any(term in description for term in ["biology", "chemistry", "mathematics", "environmental"]):
            return "Natural Sciences & Mathematics"
        elif any(term in description for term in ["computer", "information", "data"]):
            return "Computer Studies"
        elif any(term in description for term in ["economics", "political", "psychology", "sociology", "anthropology", "social"]):
            return "Social Sciences"
        elif any(term in description for term in ["accountancy", "accounting"]):
            return "Accountancy"
        elif any(term in description for term in ["business", "entrepreneurship", "finance", "marketing", "management", "public"]):
            return "Business Management"
        elif any(term in description for term in ["education", "elementary", "secondary", "childhood"]):
            return "Education"
        elif any(term in description for term in ["engineering", "architecture"]):
            return "Engineering & Architecture"
        elif "nursing" in description:
            return "Nursing"
        else:
            return "Unknown Cluster"

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
        Uses configurable school and cluster keywords from config
        """
        query_lower = query.lower()
        
        # Normalize school abbreviations for better detection
        normalized_query = self._normalize_school_abbreviations(query_lower)
        
        # Detect query type using configurable keywords
        if 'cluster' in query_lower:
            return self._extract_cluster_info_configurable(program_list_content, normalized_query)
        elif self._detect_school_query_enhanced(query_lower, normalized_query):
            return self._extract_school_info_configurable(program_list_content, normalized_query)
        else:
            # General program list
            return f"Here are the available undergraduate programs at Ateneo de Davao University:\n\n{program_list_content}"

    def _detect_school_query(self, query_lower: str) -> bool:
        """Detect if query is asking for school-specific information using config"""
        school_keywords = self._get_school_keywords()
        return any(keyword in query_lower for keyword in school_keywords.keys())
    
    def _detect_school_query_enhanced(self, query_lower: str, normalized_query: str) -> bool:
        """Enhanced school query detection that handles abbreviations"""
        # Check original query
        if self._detect_school_query(query_lower):
            return True
        
        # Check normalized query
        if self._detect_school_query(normalized_query):
            return True
        
        # Check school abbreviations directly
        school_abbreviations = self._get_school_abbreviations()
        for abbrev in school_abbreviations.keys():
            if abbrev.lower() in query_lower:
                return True
        
        return False

    def _extract_cluster_info_configurable(self, content: str, query: str) -> str:
        """Extract cluster-specific information using configurable keywords"""
        lines = content.split('\n')
        result = []
        in_target_cluster = False
        current_school = None
        
        # Get cluster keywords from config
        cluster_keywords = self._get_cluster_keywords()
        
        # Find target cluster from query
        target_cluster = None
        for keyword, cluster_name in cluster_keywords.items():
            if keyword in query:
                target_cluster = cluster_name
                break
        
        if not target_cluster:
            return "I couldn't identify which cluster you're asking about. Please specify a cluster like 'computer', 'business', 'engineering', etc."
        
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
        
        return '\n'.join(result) if result else f"I couldn't find programs in the {target_cluster} cluster."

    def _extract_school_info_configurable(self, content: str, query: str) -> str:
        """Extract school-specific information using configurable keywords"""
        lines = content.split('\n')
        result = []
        in_target_school = False
        
        # Get school keywords from config
        school_keywords = self._get_school_keywords()
        
        # Find target school from query
        target_school = None
        for keyword, school_name in school_keywords.items():
            if keyword in query:
                target_school = school_name
                break
        
        if not target_school:
            return "I couldn't identify which school you're asking about. Please specify a school like 'arts', 'business', 'engineering', etc."
        
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
        
        return '\n'.join(result) if result else f"I couldn't find information about {target_school}."

    def _preprocess_pronoun_query(self, query: str, program_context: str = None) -> str:
        """
        Preprocess queries with pronouns to avoid misinterpretation.
        Enhanced to handle subject mapping queries with context tracking.
        """
        query_lower = query.lower().strip()
        
        # Enhanced detection for subject mapping queries
        subject_mapping_indicators = [
            'programming subjects', 'math subjects', 'mathematics subjects',
            'science subjects', 'english subjects', 'business subjects',
            'subjects', 'courses', 'programming courses', 'math courses',
            'coding subjects', 'technical subjects'
        ]
        
        curriculum_indicators = ['curriculum', 'program structure', 'course outline']
        
        is_subject_mapping = any(indicator in query_lower for indicator in subject_mapping_indicators)
        is_curriculum = any(indicator in query_lower for indicator in curriculum_indicators)
        
        # Determine context source based on query type
        if is_subject_mapping:
            context_source = "subject_mapping"
            subject_type = self._extract_subject_type_from_query(query)
            print(f"🔍 Detected pronoun in subject mapping query: '{query}' (subject_type: {subject_type})")
        elif is_curriculum:
            context_source = "curriculum"
            print(f"📚 Detected pronoun in curriculum query: '{query}'")
        else:
            context_source = "general"
            print(f"🔄 Detected pronoun in general query: '{query}'")
        
        if not program_context:
            return query
        
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

    def _classify_programs_query_intent(self, query: str) -> str:
        """
        Classify programs query intent using semantic similarity (TF-IDF + Word2Vec).
        Handles typos naturally by leveraging existing hybrid infrastructure.
        
        Returns: 'subject_mapping', 'curriculum', 'overview', or 'mixed'
        
        Note: 'curriculum' intent includes all program-specific queries (curriculum, subjects, program details)
        and always starts with 1st year curriculum display.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        print(f"🧠 Classifying programs query intent using semantic similarity: '{query}'")
        
        # Training examples for each intent type (including typo variations)
        training_examples = {
            'subject_mapping': [
                "what programs have math",
                "which programs offer calculus", 
                "programs with programming",
                "what programs teach statistics",
                "programs that don't offer OJT",
                "what programs do not have thesis",
                "what math subjects are in BS CS",
                "how many programming courses in BS IT",
                # Additional variations for better matching
                "what are the programs that offer OJT",
                "what are the programs that have math",
                "what are the programs that teach programming",
                "which are the programs that offer calculus",
                "programs that offer thesis",
                "programs that provide internship",
                "what programs offer practicum",
                # Typo variations
                "waht programs hav math",
                "wich programs ofer calculus",
                "programs with programing",
                "waht programs dont offer ojt",
                "wat are the programs that ofer ojt",
                # Program-specific subject queries - ADDED FOR BETTER CLASSIFICATION
                "what are the programming subjects in Information Systems",
                "what are the thesis subjects in IT", 
                "what are the math subjects in Computer Science",
                "what programming subjects are in Data Science",
                "what thesis subjects are in Engineering"
            ],
            'curriculum': [
                # Original curriculum queries
                "what is the curriculum for BS CS",
                "show me the curriculum of BS IT", 
                "curriculum for computer science",
                "what about 2nd year",
                "can you proceed with remaining",
                
                # Former "specific" queries - now part of curriculum
                "tell me about BS CS program",
                "what do you study in computer science",
                "program requirements for BS IT",
                "what is BS CS",
                "BS IT program details",
                "information about nursing program",
                "describe computer science program",
                "program structure for engineering",
                
                # Typo variations for all above
                "waht is the curriculun for BS CS",
                "shwo me the curriculum",
                "tel me about BS CS program",
                "waht do you study in compter science",
                "program requirments for BS IT",
                "waht is BS CS",
                "informaton about nursing program"
            ],
            'overview': [
                "what programs are available",
                "list all programs",
                "show me all courses",
                "what can I study here",
                "programs offered by ADDU",
                "available degrees",
                # More specific overview patterns (to prevent misclassification)
                "what are the programs available in ateneo",
                "what programs are in ateneo",
                "what are the programs in ateneo de davao",
                "show me all programs at ateneo",
                "list programs at ateneo",
                "what programs does ateneo offer",
                "what programs are offered at ateneo",
                "available programs at ateneo",
                # General variations
                "what are the available programs",
                "what programs can I study",
                "what are all the programs",
                "show me all available programs",
                # Typo variations
                "waht programs ar available",
                "lst all programs",
                "shwo me all courses",
                "wat are the programs in ateneo"
            ]
        }
        
        # Flatten training data
        all_examples = []
        labels = []
        for intent, examples in training_examples.items():
            all_examples.extend(examples)
            labels.extend([intent] * len(examples))
        
        try:
            # Vectorize all examples using existing hybrid infrastructure
            query_vector = self._vectorize_query(query)
            example_vectors = []
            
            for example in all_examples:
                try:
                    example_vector = self._vectorize_query(example)
                    example_vectors.append(example_vector)
                except Exception as e:
                    print(f"⚠️ Error vectorizing example '{example}': {e}")
                    # Create zero vector as fallback
                    example_vectors.append(np.zeros_like(query_vector))
            
            if not example_vectors:
                print("❌ No example vectors created, falling back to mixed intent")
                return "mixed"
            
            # Calculate similarities using cosine similarity
            similarities = []
            for example_vector in example_vectors:
                try:
                    similarity = cosine_similarity(
                        query_vector.reshape(1, -1), 
                        example_vector.reshape(1, -1)
                    )[0][0]
                    similarities.append(similarity)
                except Exception as e:
                    print(f"⚠️ Error calculating similarity: {e}")
                    similarities.append(0.0)
            
            # Find best match
            if similarities:
                best_match_idx = np.argmax(similarities)
                best_similarity = similarities[best_match_idx]
                best_intent = labels[best_match_idx]
                
                print(f"🎯 Best match: '{all_examples[best_match_idx]}' (similarity: {best_similarity:.3f}) → {best_intent}")
                
                # Confidence threshold for classification
                similarity_threshold = 0.4  # Configurable threshold
                if best_similarity > similarity_threshold:
                    print(f"✅ Intent classified as '{best_intent}' with confidence {best_similarity:.3f}")
                    return best_intent
                else:
                    print(f"⚠️ Low confidence ({best_similarity:.3f} < {similarity_threshold}), defaulting to mixed")
                    return "mixed"
            else:
                print("❌ No similarities calculated, falling back to mixed intent")
                return "mixed"
                
        except Exception as e:
            print(f"❌ Error in semantic intent classification: {e}")
            # Fallback to simple keyword matching
            return self._classify_query_intent_fallback(query)
    
    def _classify_query_intent_fallback(self, query: str) -> str:
        """
        Fallback intent classification using simple keyword matching.
        Used when semantic similarity fails.
        """
        query_lower = query.lower()
        
        # Simple keyword-based classification as fallback
        subject_mapping_keywords = ['programs', 'have', 'offer', 'teach', 'with', 'without', 'subjects', 'courses']
        # Expanded curriculum keywords to include former "specific" patterns
        curriculum_keywords = ['curriculum', 'subjects', 'study', 'learn', 'program', 'year', 'about', 'details', 'requirements', 'structure', 'describe', 'information']
        overview_keywords = ['available', 'list', 'all', 'show', 'what programs']
        
        subject_score = sum(1 for keyword in subject_mapping_keywords if keyword in query_lower)
        curriculum_score = sum(1 for keyword in curriculum_keywords if keyword in query_lower)
        overview_score = sum(1 for keyword in overview_keywords if keyword in query_lower)
        
        if subject_score > curriculum_score and subject_score > overview_score:
            return "subject_mapping"
        elif curriculum_score > overview_score:
            return "curriculum"
        elif overview_score > 0:
            return "overview"
        else:
            return "mixed"

    def _classify_query_intent(self, query: str) -> str:
        """
        Classify query intent as 'overview', 'specific', 'mixed', or 'subject_mapping'
        
        Overview queries: Ask for program lists, available programs, school overviews
        Specific queries: Ask for curriculum, subjects, program details
        Subject mapping queries: Ask which programs have/offer specific subjects
        Mixed queries: Ambiguous or contain both types of indicators
        """
        import re
        
        query_lower = query.lower()
        
        # Subject mapping indicators - queries asking which programs have specific subjects
        # or what subjects are in specific programs
        subject_mapping_patterns = [
            # General subject mapping patterns (which programs have X)
            r'\bwhat\s+programs?\s+(have|offer|include|contain|teach)\s+',
            r'\bwhich\s+programs?\s+(have|offer|include|contain|teach)\s+',
            r'\bprograms?\s+(with|that\s+have|that\s+offer|that\s+include)\s+',
            r'\bunder\s+.+cluster.+what\s+programs?\s+(have|offer)\s+',
            
            # Enhanced negative subject mapping patterns (which programs don't have X)
            r'\bwhat\s+programs?\s+(do\s+not|don\'t|doesn\'t|doesnt)\s+(have|offer|include|provide)\s+',
            r'\bwhich\s+programs?\s+(do\s+not|don\'t|doesn\'t|doesnt)\s+(have|offer|include|provide)\s+',
            r'\bprograms?\s+(without|that\s+do\s+not|that\s+don\'t|that\s+doesn\'t|that\s+doesnt)\s+',
            r'\bprograms?\s+that\s+(do\s+not|don\'t|doesn\'t|doesnt)\s+(offer|have|include|provide)\s+',
            r'\bwhat\s+are\s+the\s+programs?\s+that\s+(do\s+not|don\'t|doesn\'t|doesnt)\s+(offer|have)\s+',
            r'\bprograms?\s+that\s+(doesn\'t|doesnt)\s+(offer|have|include|provide)\s+',
            
            # Program-specific subject queries (what X subjects are in program Y)
            r'\bwhat\s+are\s+the\s+.+(?:subjects?|courses?)\s+(in|under|for)\s+',
            r'\blist\s+.+(?:subjects?|courses?)\s+(in|under|for)\s+',
            r'\bwhat\s+(?:subjects?|courses?)\s+(?:are\s+)?(in|does|offered\s+in|offered\s+by)\s+',
            r'\bwhat\s+(?:subjects?|courses?)\s+(?:are\s+)?(taught|available|included)\s+in\s+',
            r'\bwhat\s+\w+\s+(?:subjects?|courses?)\s+(?:are\s+)?(?:in|offered\s+in|taught\s+in)\s+',
            r'\bwhat\s+\w+\s+(?:courses?)\s+(?:does)\s+\w+\s+(?:offer|have)\s*',
            r'\bwhat\s+\w+\s+(?:subjects?|courses?)\s+are\s+in\s+',
            r'\bwhat\s+\w+\s+(?:courses?)\s+does\s+\w+\s+\w+\s+offer\b',
            
            # Count queries (how many X subjects in program Y)
            r'\bhow\s+many\s+.+(?:subjects?|courses?)\s+(?:in|does)\s+',
            
            # Direct program-specific subject queries
            r'\b(?:subjects?|courses?)\s+(?:in|offered\s+in|offered\s+by)\s+',
            r'\b(?:programming|math|science|engineering|business|research)\s+(?:subjects?|courses?)\s+(?:in|offered\s+in)\s+'
        ]
        
        # Check for subject mapping patterns first (highest priority)
        has_subject_mapping = False
        matched_pattern = None
        for i, pattern in enumerate(subject_mapping_patterns):
            if re.search(pattern, query_lower):
                has_subject_mapping = True
                matched_pattern = f"Pattern {i+1}: {pattern}"
                print(f"🎯 Subject mapping pattern matched: {matched_pattern}")
                break
        
        # Additional check for common subject mapping queries that might be missed
        if not has_subject_mapping:
            # Check for "what are the programs that [verb] [subject]" pattern
            additional_subject_mapping = re.search(r'\bwhat\s+are\s+the\s+programs?\s+that\s+(offer|have|teach|include|contain|provide)\s+\w+', query_lower)
            if additional_subject_mapping:
                has_subject_mapping = True
                matched_pattern = f"Additional pattern: what are the programs that [verb] [subject]"
                print(f"🎯 Subject mapping pattern matched: {matched_pattern}")
        
        # Special check for program-specific subject queries that might not match the patterns
        # For example: "what are the programming courses offered in BS CS"
        if not has_subject_mapping:
            # Check for program-specific subject queries with program name
            program_subject_pattern = r'\b(?:what|which|show|list)\s+(?:are\s+)?(?:the\s+)?([a-z]+)\s+(?:subjects?|courses?)\s+(?:offered|available|taught|included)\s+(?:in|by|at|under)\s+([a-z0-9\s]+)'
            program_match = re.search(program_subject_pattern, query_lower)
            if program_match:
                subject_type = program_match.group(1)  # e.g., "programming"
                program_name = program_match.group(2)  # e.g., "BS CS"
                
                # Check if subject_type is a valid subject category
                subject_categories = ['programming', 'math', 'science', 'research', 'business', 
                                     'engineering', 'language', 'english', 'communication', 'thesis',
                                     'capstone', 'project', 'practicum', 'internship', 'ojt']
                
                if subject_type in subject_categories or 'program' in query_lower:
                    print(f"🎯 Detected program-specific subject query: '{subject_type}' in '{program_name}'")
                    has_subject_mapping = True
        
        if has_subject_mapping:
            return "subject_mapping"
        
        # Curriculum-specific intent detection patterns
        curriculum_patterns = [
            r'\bcurriculum\b',
            r'\bwhat\s+(?:are\s+the\s+)?(?:subjects?|courses?)\s+(?:in|for|of)\s+',
            r'\bwhat\s+(?:about|is)\s+(?:the\s+)?(?:curriculum|subjects?|courses?)\s+(?:of|for|in)\s+',
            r'\bshow\s+(?:me\s+)?(?:the\s+)?(?:curriculum|subjects?|courses?)\s+(?:of|for|in)\s+',
            r'\blist\s+(?:the\s+)?(?:curriculum|subjects?|courses?)\s+(?:of|for|in)\s+',
            r'\btell\s+me\s+about\s+(?:the\s+)?(?:curriculum|subjects?|courses?)\s+(?:of|for|in)\s+',
            # Follow-up patterns
            r'\bwhat\s+about\s+(?:the\s+)?(?:\d+(?:st|nd|rd|th)\s+year|year\s+\d+)\b',
            r'\bwhat\s+about\s+(?:second|third|fourth|2nd|3rd|4th)\s+year\b',
            r'\bcan\s+you\s+proceed\s+with\s+(?:the\s+)?remaining\b',
            r'\bwhat\s+(?:are\s+)?(?:the\s+)?(?:next|remaining)\b',
            r'\bcontinue\s+with\s+(?:the\s+)?(?:curriculum|subjects?|courses?)\b',
            r'\bshow\s+(?:me\s+)?(?:the\s+)?(?:next|remaining)\s+(?:year|subjects?|courses?)\b',
            r'\bwhat\'?s\s+next\b',
            r'\bshow\s+(?:me\s+)?(?:the\s+)?remaining\s+years?\b'
        ]
        
        # Check for curriculum patterns
        has_curriculum = False
        matched_curriculum_pattern = None
        for i, pattern in enumerate(curriculum_patterns):
            if re.search(pattern, query_lower):
                has_curriculum = True
                matched_curriculum_pattern = f"Curriculum Pattern {i+1}: {pattern}"
                print(f"📚 Curriculum pattern matched: {matched_curriculum_pattern}")
                break
        
        if has_curriculum:
            return "curriculum"
        
        # Overview indicators - queries asking for program lists or availability
        overview_patterns = [
            r'\b(what|show|list|all)\s+(programs?|degrees?)\s+(are\s+)?(available|offered|in|under)\b',
            r'\bprograms?\s+(in|under|at|offered by)\s+(school|college|addu)\b',
            r'\b(available|offered)\s+programs?\b',
            r'\bwhat\s+(does|are)\s+(school|college|addu)\s+(offer|have)\b',
            r'\bshow\s+me\s+all\s+programs?\b',
            r'\blist\s+(of\s+)?(all\s+)?programs?\b',
            r'\bwhat\s+programs?\s+(does|do|are)\s+',
            r'\bprograms?\s+(available|offered)\s+(in|at|under)\b',
            r'\ball\s+(the\s+)?programs?\s+(in|under|at)\b',
            r'\bwhich\s+programs?\s+(are\s+)?(available|offered)\b'
        ]
        
        # Specific program indicators - queries asking for curriculum, subjects, details
        specific_patterns = [
            r'\b(curriculum|subjects?|courses?|syllabus)\s+(for|of|in)\s+',
            r'\b(what|show)\s+(subjects?|courses?|curriculum)\s+(are\s+)?(in|for|of)\s+',
            r'\b(program|degree)\s+(requirements?|details?)\b',
            r'\b(study|academic)\s+plan\b',
            r'\bwhat\s+(subjects?|courses?)\s+(are\s+)?(in|for|of|required)\s+',
            r'\bshow\s+me\s+(the\s+)?(curriculum|subjects?|courses?)\b',
            r'\b(course|subject)\s+(list|outline)\b',
            r'\bprogram\s+(structure|content)\b',
            r'\bwhat\s+(do\s+)?(you|we)\s+(study|learn)\s+in\s+',
            r'\bsyllabus\s+(for|of)\s+'
        ]
        
        # Check for pattern matches
        has_overview = any(re.search(pattern, query_lower) for pattern in overview_patterns)
        has_specific = any(re.search(pattern, query_lower) for pattern in specific_patterns)
        
        # Additional context-based classification
        # Overview context words
        overview_context = ['school', 'college', 'university', 'addu', 'available', 'offered', 'all', 'list']
        # Specific context words  
        specific_context = ['curriculum', 'subjects', 'courses', 'syllabus', 'requirements', 'study', 'learn']
        
        overview_context_count = sum(1 for word in overview_context if word in query_lower)
        specific_context_count = sum(1 for word in specific_context if word in query_lower)
        
        # Decision logic
        if has_overview and not has_specific:
            return "overview"
        elif has_specific and not has_overview:
            return "specific"
        elif overview_context_count > specific_context_count:
            return "overview"
        elif specific_context_count > overview_context_count:
            return "specific"
        else:
            return "mixed"

    def _detect_cluster_query(self, query: str) -> str:
        """
        Detect if query is asking for cluster-specific information
        Returns: cluster name if detected, None otherwise
        """
        query_lower = query.lower().strip()
        
        # First check for school-level queries that should include all clusters
        school_abbreviations = self._get_school_abbreviations()
        for abbrev, school_name in school_abbreviations.items():
            if abbrev.lower() in query_lower or school_name.lower() in query_lower:
                # This is a school-level query, not a cluster query
                return None
        
        # Get cluster keywords from config
        cluster_keywords = self._get_cluster_keywords()
        
        # Check if query contains cluster-related keywords
        for keyword, cluster_name in cluster_keywords.items():
            if keyword in query_lower:
                return cluster_name
        
        return None

    def _get_cluster_keywords(self) -> dict:
        """Get cluster keywords from config"""
        if not hasattr(self, '_cluster_keywords_cache'):
            try:
                with open(self._config_file_path, 'r') as f:
                    config = json.load(f)
                    self._cluster_keywords_cache = config.get('cluster_keywords', {})
            except Exception as e:
                print(f"⚠️ Error loading cluster keywords: {e}")
                self._cluster_keywords_cache = {}
        return self._cluster_keywords_cache

    def _get_school_abbreviations(self) -> dict:
        """Get school abbreviations from config"""
        if not hasattr(self, '_school_abbreviations_cache'):
            try:
                with open(self._config_file_path, 'r') as f:
                    config = json.load(f)
                    self._school_abbreviations_cache = config.get('school_abbreviations', {})
            except Exception as e:
                print(f"⚠️ Error loading school abbreviations: {e}")
                self._school_abbreviations_cache = {}
        return self._school_abbreviations_cache

    def _get_programs_in_cluster(self, cluster_name: str) -> List[str]:
        """Get all programs that belong to a specific cluster from JSON config"""
        try:
            with open(self._config_file_path, 'r') as f:
                config = json.load(f)
                abbreviations = config.get('abbreviations', {})
                
                cluster_programs = []
                for abbrev, data in abbreviations.items():
                    if data.get('cluster', '').lower() == cluster_name.lower():
                        # Get the full program name for filename matching
                        full_name = data.get('full_name', '')
                        if full_name:
                            cluster_programs.append(full_name)
                        # Also add the abbreviation for matching
                        cluster_programs.append(abbrev)
                
                return cluster_programs
        except Exception as e:
            print(f"⚠️ Error loading programs for cluster {cluster_name}: {e}")
            return []

    def retrieve_documents_with_intent_classification(self, query: str, topic_id: str, top_k: int = 2) -> List[Dict]:
        """
        ENHANCED RETRIEVAL WITH INTENT CLASSIFICATION:
        1. Classify query intent (overview, specific, mixed)
        2. Apply intent-aware document filtering
        3. Use existing TF-IDF/Word2Vec hybrid scoring
        4. Apply intent-based score boosting
        """
        import re
        
        # Step 1: Normalize school abbreviations first
        school_normalized_query = self._normalize_school_abbreviations(query)
        if school_normalized_query != query:
            print(f"📝 School normalized query: '{query}' → '{school_normalized_query}'")
            query = school_normalized_query
        
        # Step 2: Classify query intent
        intent = self._classify_programs_query_intent(query)
        print(f"🎯 Query intent classified as: '{intent}'")
        
        # Step 3: Detect cluster query and apply cluster filtering
        cluster_filter = self._detect_cluster_query(query)
        if cluster_filter:
            print(f"🎯 Detected cluster query: '{cluster_filter}'")
        
        # Step 4: Get base results using existing hybrid method with cluster filtering
        base_results = self.retrieve_documents_by_topic_hybrid(query, topic_id, top_k * 3, cluster_filter=cluster_filter)  # Get more to filter
        
        if not base_results:
            print("⚠️ No base results found, falling back to simple retrieval")
            return self.retrieve_documents_by_topic_keywords_simple(query, topic_id, top_k)
        
        # Step 3: Apply intent-based filtering and scoring
        enhanced_results = []
        
        for doc in base_results:
            doc_metadata = doc.get('metadata', {})
            doc_type = doc.get('document_type', '')
            filename = doc.get('filename', '').lower()
            
            # Get base score (from existing TF-IDF/Word2Vec hybrid)
            base_score = doc.get('relevance', 0.0)
            
            # Apply intent-based scoring
            intent_multiplier = 1.0
            
            if intent == "overview":
                # Boost directory/overview documents
                if any(keyword in filename for keyword in ['directory', 'programs', 'list', 'overview']):
                    intent_multiplier = 1.5
                    print(f"📈 Overview boost for: {filename[:50]}...")
                elif doc_type == 'program_directory':
                    intent_multiplier = 1.5
                    print(f"📈 Directory document boost for: {filename[:50]}...")
                # Slightly penalize very specific curriculum documents
                elif any(keyword in filename for keyword in ['curriculum', 'syllabus', 'course']):
                    intent_multiplier = 0.8
                    
            elif intent == "specific":
                # Boost curriculum/specific documents
                if any(keyword in filename for keyword in ['curriculum', 'syllabus', 'course', 'program']):
                    intent_multiplier = 1.5
                    print(f"📈 Specific boost for: {filename[:50]}...")
                elif doc_type == 'program_curriculum':
                    intent_multiplier = 1.5
                    print(f"📈 Curriculum document boost for: {filename[:50]}...")
                # Slightly penalize very general directory documents
                elif any(keyword in filename for keyword in ['directory', 'list', 'overview']):
                    intent_multiplier = 0.8
            
            # Mixed intent uses normal scoring (no boost/penalty)
            
            # Calculate final score
            final_score = base_score * intent_multiplier
            
            enhanced_results.append({
                'id': doc.get('id'),
                'content': doc.get('content'),
                'metadata': doc_metadata,
                'filename': doc.get('filename', ''),
                'document_type': doc.get('document_type', ''),
                'relevance': final_score,
                'base_score': base_score,
                'intent_multiplier': intent_multiplier,
                'intent': intent
            })
        
        # Step 4: Sort by enhanced score and return top_k
        enhanced_results.sort(key=lambda x: x['relevance'], reverse=True)
        final_results = enhanced_results[:top_k]
        
        # Log the scoring details
        print(f"🎯 Intent-enhanced results (top {len(final_results)}):")
        for i, doc in enumerate(final_results, 1):
            filename = doc.get('filename', 'Unknown')[:50]
            print(f"   {i}. {filename}...")
            print(f"      Final Score: {doc['relevance']:.3f}")
            print(f"      Base Score: {doc['base_score']:.3f}")
            print(f"      Intent Multiplier: {doc['intent_multiplier']:.1f}x")
        
        return final_results

def test_fast_hybrid_chatbot_together():
    """Test the fast hybrid chatbot with Together AI"""
    chatbot = FastHybridChatbotTogether(use_chroma=True)
    
    print("\n⚡🔍 FAST HYBRID CHATBOT WITH TOGETHER AI ⚡🔍")
    print("Type 'exit' to quit\n")

    max_tokens = 1024  # Higher default for complete responses with 70B model
