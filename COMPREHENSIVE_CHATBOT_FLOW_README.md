# ADDU Admissions Chatbot - Comprehensive Technical Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Components](#architecture-components)
3. [Topic-Specific Retrieval Systems](#topic-specific-retrieval-systems)
4. [Dialogue History & Context Management](#dialogue-history--context-management)
5. [Hybrid Retrieval Implementation](#hybrid-retrieval-implementation)
6. [Response Generation & Prompt Engineering](#response-generation--prompt-engineering)
7. [Admin System & Document Processing](#admin-system--document-processing)
8. [Database Schema & Models](#database-schema--models)
9. [API Endpoints & Request Flow](#api-endpoints--request-flow)
10. [Performance Optimization](#performance-optimization)

## System Overview

The ADDU Admissions Chatbot is a sophisticated guided conversation system that uses a hybrid retrieval-augmented generation (RAG) architecture. It combines multiple NLP techniques with cloud-based LLM integration to provide accurate, context-aware responses about university admissions, programs, and fees through structured topic-based interactions.

### Core Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                               │
│                       (React Frontend)                               │
│                                                                       │
│                     ┌──────────────────┐                            │
│                     │  Guided Chat     │                            │
│                     │  Interface       │                            │
│                     │                  │                            │
│                     │ + Topic Selection│                            │
│                     │ + Conversation   │                            │
│                     │ + History View   │                            │
│                     └──────────────────┘                            │
│                              │                                       │
└──────────────────────────────┼───────────────────────────────────────┘
                               │
                               │ HTTP POST
                               │ /chatbot/chat/guided/
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DJANGO BACKEND                                  │
│                                                                       │
│                    ┌─────────────────────────────┐                  │
│                    │  Guided Conversation        │                  │
│                    │  System                     │                  │
│                    │                             │                  │
│                    │ + Topic Selection           │                  │
│                    │ + Intent Classification     │                  │
│                    │ + Document Filtering        │                  │
│                    │ + Context Management        │                  │
│                    │ + Dialogue History          │                  │
│                    │                             │                  │
│                    │ + TF-IDF Retrieval          │                  │
│                    │ + Word2Vec Vectors          │                  │
│                    │ + ChromaDB Search           │                  │
│                    └─────────────────────────────┘                  │
│                                 │                                    │
└─────────────────────────────────┼────────────────────────────────────┘
                                  │
                                  ▼
                    ┌───────────────────────┐
                    │  TOGETHER AI LLM      │
                    │  (Cloud Service)      │
                    │                       │
                    │  Llama-4-Scout-17B    │
                    │  Streaming Response   │
                    │  Temperature: 0.3     │
                    └───────────────────────┘
```

### Key Features

- **Guided Conversation Flow**: Topic-based conversation management with structured interactions
- **Hybrid Retrieval**: TF-IDF + Word2Vec + ChromaDB semantic search
- **Intent Classification**: Semantic similarity-based query classification
- **Dialogue History Tracking**: Multi-turn conversation continuity with context preservation
- **Topic-Specific Retrieval**: Specialized document retrieval for admissions, programs, and fees
- **Admin Document Management**: Full CRUD operations for documents and folders
- **Real-time Streaming**: Together AI integration with streaming responses
- **Multi-modal Data**: Supports PDF, Excel, CSV, and text documents

## Topic-Specific Retrieval Systems

### 1. Admissions & Enrollment Retrieval

The admissions retrieval system is optimized for detecting student types and requirement-specific queries.

```python
def retrieve_admissions_documents(self, query: str, top_k: int = 2) -> List[Dict]:
    """
    Specialized retrieval for admissions, enrollment, requirements, and documents.
    Optimized for detecting student types and requirement-specific queries.
    """
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

        # Score documents with admissions-specific logic
        scored_results = []
        for i, (doc_id, content, metadata) in enumerate(zip(all_ids, all_contents, all_metadatas)):
            # Calculate specialized scores
            filename_score = self._calculate_admissions_filename_score(
                metadata.get('filename', '').lower(), student_type, requirement_type
            )
            keyword_score = self._calculate_keyword_score(
                metadata.get('keywords', '').lower(), query.lower()
            )
            content_score = self._calculate_admissions_content_score(
                content.lower(), query.lower(), student_type, requirement_type
            )

            # Apply strategy priorities
            specialized_score = (
                filename_score * priorities.get('filename', 0.3) +
                keyword_score * priorities.get('keywords', 0.4) +
                content_score * priorities.get('content', 0.3)
            )

            scored_results.append({
                'id': doc_id,
                'content': content,
                'metadata': metadata,
                'relevance': specialized_score,
                'student_type': student_type,
                'requirement_type': requirement_type
            })

        # Sort by specialized score and return top_k
        scored_results.sort(key=lambda x: x['relevance'], reverse=True)
        return scored_results[:top_k]

    except Exception as e:
        print(f"❌ Admissions retrieval error: {e}")
        return []
```

#### Student Type Detection

```python
def _detect_student_type(self, query: str) -> str:
    """Detect student type from query for specialized retrieval"""
    query_lower = query.lower()

    # Transfer student indicators
    transfer_indicators = [
        'transfer', 'transferee', 'shifter', 'lateral entry',
        'changing course', 'shift', 'transferring'
    ]

    # International student indicators
    international_indicators = [
        'international', 'foreign', 'overseas', 'non-filipino',
        'visa', 'passport', 'embassy'
    ]

    # Scholar student indicators
    scholar_indicators = [
        'scholar', 'scholarship', 'financial aid', 'grant',
        'discount', 'free tuition'
    ]

    if any(indicator in query_lower for indicator in transfer_indicators):
        return 'transfer'
    elif any(indicator in query_lower for indicator in international_indicators):
        return 'international'
    elif any(indicator in query_lower for indicator in scholar_indicators):
        return 'scholar'
    else:
        return 'new'  # Default to new student
```

#### Requirement Type Detection

```python
def _detect_requirement_type(self, query: str) -> str:
    """Detect requirement type from query"""
    query_lower = query.lower()

    document_indicators = [
        'document', 'requirements', 'papers', 'forms',
        'certificate', 'transcript', 'diploma'
    ]

    process_indicators = [
        'process', 'procedure', 'steps', 'how to',
        'application', 'enrollment'
    ]

    deadline_indicators = [
        'deadline', 'when', 'date', 'schedule',
        'timeline', 'period'
    ]

    if any(indicator in query_lower for indicator in document_indicators):
        return 'documents'
    elif any(indicator in query_lower for indicator in process_indicators):
        return 'process'
    elif any(indicator in query_lower for indicator in deadline_indicators):
        return 'deadlines'
    else:
        return 'general'
```

### 2. Programs & Courses Retrieval

The programs retrieval system handles curriculum queries, program availability, and course information with sophisticated program matching.

```python
def retrieve_programs_documents(self, query: str, top_k: int = 2) -> List[Dict]:
    """
    Specialized retrieval for programs, courses, and curriculum.
    Uses flexible approach: get all documents, then filter by keywords OR document_type.
    """
    print(f"📚 Programs-specialized retrieval for: '{query}'")

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

        # Step 2: Also get ALL documents that might have program keywords
        all_docs_by_keywords = collection.get(
            where={"source": "pdf_scrape"},
            include=["documents", "metadatas"]
        )

        # Step 3: Combine and deduplicate documents
        combined_docs = self._combine_and_deduplicate_docs(all_docs, all_docs_by_keywords)

        # Step 4: Score documents with programs-specific logic
        scored_results = []
        for doc_data in combined_docs:
            metadata = doc_data['metadata']
            content = doc_data['content']

            # Calculate specialized scores
            filename_score = self._calculate_programs_filename_score(
                metadata.get('filename', '').lower(), program_info
            )
            keyword_score = self._calculate_keyword_score(
                metadata.get('keywords', '').lower(), query.lower()
            )
            content_score = self._calculate_programs_content_score(
                content.lower(), query.lower(), program_info
            )

            # Apply strategy priorities
            specialized_score = (
                filename_score * priorities.get('filename', 0.3) +
                keyword_score * priorities.get('keywords', 0.4) +
                content_score * priorities.get('content', 0.3)
            )

            scored_results.append({
                'id': doc_data['id'],
                'content': content,
                'metadata': metadata,
                'relevance': specialized_score,
                'program_info': program_info
            })

        # Sort by specialized score and return top_k
        scored_results.sort(key=lambda x: x['relevance'], reverse=True)
        return scored_results[:top_k]

    except Exception as e:
        print(f"❌ Programs retrieval error: {e}")
        return []
```

#### Program Information Extraction (JSON-Based Configuration)

The system uses a sophisticated JSON-based configuration system for program information extraction:

```python
def _extract_program_info(self, query: str) -> Dict:
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
    abbreviations = self._get_program_abbreviations()  # Loads from JSON

    matches = []

    # Find all potential program matches using JSON configuration
    for abbrev, abbrev_data in abbreviations.items():
        if not isinstance(abbrev_data, dict):
            continue

        full_name = abbrev_data.get("full_name", "")
        description = abbrev_data.get("description", "")
        priority = abbrev_data.get("priority", "safe")
        is_common_word = abbrev_data.get("is_common_word", False)

        # Calculate match score using sophisticated scoring
        match_score = self._calculate_program_match_score(query_lower, abbrev, full_name, description)

        if match_score > 0:
            # Apply priority filtering based on JSON configuration
            should_include = False

            if priority == "safe":
                should_include = True
            elif priority == "context_aware":
                # Check for program context using JSON patterns
                context_patterns = self._get_context_patterns()
                program_keywords = context_patterns.get("program_keywords", [])
                has_program_context = any(re.search(rf'\b{keyword}\b', query_lower) for keyword in program_keywords)
                should_include = has_program_context
            elif priority == "context_required":
                # Only include with strong context (for common English words)
                context_patterns = self._get_context_patterns()
                strong_program_patterns = context_patterns.get("strong_program", [])
                has_strong_context = any(re.search(pattern, query_lower) for pattern in strong_program_patterns)
                should_include = has_strong_context

            if should_include:
                matches.append({
                    'abbrev': abbrev,
                    'full_name': full_name,
                    'description': description,
                    'priority': priority,
                    'match_score': match_score,
                    'school': abbrev_data.get('school', ''),
                    'cluster': abbrev_data.get('cluster', '')
                })

    # Sort matches by score and priority
    matches.sort(key=lambda x: (x['match_score'], x['priority'] == 'safe'), reverse=True)

    if matches:
        best_match = matches[0]
        program_info['program_name'] = best_match['full_name']
        program_info['course_code'] = best_match['abbrev']
        program_info['context_source'] = 'json_config'

    # Extract year level and curriculum context
    curriculum_terms = ['curriculum', 'subjects', 'courses', 'year', 'semester']
    program_info['is_curriculum_query'] = any(term in query_lower for term in curriculum_terms)

    year_patterns = {
        'first year': 1, '1st year': 1, 'year 1': 1,
        'second year': 2, '2nd year': 2, 'year 2': 2,
        'third year': 3, '3rd year': 3, 'year 3': 3,
        'fourth year': 4, '4th year': 4, 'year 4': 4
    }

    for pattern, year in year_patterns.items():
        if pattern in query_lower:
            program_info['year_level'] = year
            break

    return program_info

def _get_program_abbreviations(self):
    """Get all program abbreviations from JSON config"""
    config = self._load_normalization_config()
    abbreviations = {}

    # Flatten all school sections into a single dict
    for school_section, programs in config["program_abbreviations"].items():
        if isinstance(programs, dict) and not school_section.startswith("_"):
            abbreviations.update(programs)

    return abbreviations

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
```

#### JSON Configuration Structure (`program_normalization_config.json`)

The system uses a comprehensive JSON configuration file that contains:

```json
{
  "program_abbreviations": {
    "_comment": "All 59 official ADDU programs with priority levels",

    "School of Arts & Sciences - Humanities & Letters": {
      "abel": {
        "full_name": "AB EL",
        "priority": "safe",
        "description": "Bachelor of Arts in English Language",
        "school": "School of Arts & Sciences",
        "cluster": "Humanities & Letters"
      },
      "eng": {
        "full_name": "AB EL",
        "priority": "context_aware",
        "description": "English Language abbreviation"
      },
      "bscs": {
        "full_name": "BS CS",
        "priority": "safe",
        "description": "Bachelor of Science in Computer Science",
        "school": "School of Arts & Sciences",
        "cluster": "Computer Studies"
      },
      "it": {
        "full_name": "BS IT",
        "priority": "context_required",
        "description": "Information Technology abbreviation",
        "is_common_word": true
      }
    }
  },

  "context_patterns": {
    "program_keywords": ["program", "course", "degree", "bachelor", "major"],
    "strong_program": ["curriculum", "subjects", "courses", "academic"],
    "problematic": ["what is it", "about it", "tell me about it"]
  },

  "common_english_words": ["it", "is", "as", "at", "an", "be"],

  "cluster_keywords": {
    "computer studies": "Computer Studies",
    "engineering": "Engineering",
    "business": "Business Management"
  },

  "school_abbreviations": {
    "sas": "School of Arts & Sciences",
    "sbg": "School of Business & Governance",
    "sea": "School of Engineering & Architecture"
  }
}
```

#### Priority System

The JSON configuration uses a sophisticated priority system:

- **`safe`**: Always normalize (e.g., "bscs" → "BS CS")
- **`context_aware`**: Only normalize with program context keywords
- **`context_required`**: Only normalize with strong program context (for common English words like "it")

This prevents false matches like interpreting "it" as "Information Technology" in casual conversation.

### 3. Fees & Payments Retrieval

The fees retrieval system handles program-specific fee calculations and payment information.

```python
def retrieve_fees_documents(self, query: str, top_k: int = 2) -> List[Dict]:
    """
    Specialized retrieval for fees, payments, and financial information.
    Optimized for program-specific fee queries and payment methods.
    """
    print(f"💰 Fees-specialized retrieval for: '{query}'")

    try:
        collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)

        # Get strategy configuration
        strategy_config = get_retrieval_strategy_config('fees_specialized')
        document_types = strategy_config.get('document_types', ['fees', 'payment'])
        priorities = strategy_config.get('metadata_priorities', {})

        # Extract fee information from query
        fee_info = self._extract_fee_info(query)
        print(f"💳 Extracted fee info: {fee_info}")

        # Get documents with fees-specific filtering
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

        # Score documents with fees-specific logic
        scored_results = []
        for i, (doc_id, content, metadata) in enumerate(zip(all_ids, all_contents, all_metadatas)):
            # Calculate specialized scores
            filename_score = self._calculate_fees_filename_score(
                metadata.get('filename', '').lower(), fee_info
            )
            keyword_score = self._calculate_keyword_score(
                metadata.get('keywords', '').lower(), query.lower()
            )
            content_score = self._calculate_fees_content_score(
                content.lower(), query.lower(), fee_info
            )

            # Apply strategy priorities
            specialized_score = (
                filename_score * priorities.get('filename', 0.3) +
                keyword_score * priorities.get('keywords', 0.4) +
                content_score * priorities.get('content', 0.3)
            )

            scored_results.append({
                'id': doc_id,
                'content': content,
                'metadata': metadata,
                'relevance': specialized_score,
                'fee_info': fee_info
            })

        # Sort by specialized score and return top_k
        scored_results.sort(key=lambda x: x['relevance'], reverse=True)
        return scored_results[:top_k]

    except Exception as e:
        print(f"❌ Fees retrieval error: {e}")
        return []
```

#### Fee Information Extraction (JSON-Based Configuration)

The fees extraction system **ALSO uses the same JSON configuration system** as programs for accurate program detection:

```python
def _extract_fee_info(self, query: str) -> dict:
    """Extract fee information from query for fees specialization"""
    query_lower = query.lower()

    fee_info = {
        'program_name': None,
        'fee_type': 'general',
        'payment_term': 'general',
        'query_category': 'general',
        'program_level': 'undergraduate'
    }

    # Fee type indicators
    fee_types = {
        'tuition': ['tuition', 'tuition fee', 'academic fee'],
        'miscellaneous': ['miscellaneous', 'misc fee', 'other fees'],
        'laboratory': ['laboratory', 'lab fee', 'lab'],
        'enrollment': ['enrollment', 'registration'],
        'total': ['total', 'overall', 'complete fee']
    }

    # Payment terms
    payment_terms = {
        'installment': ['installment', 'payment plan', 'monthly'],
        'semester': ['semester', 'per sem', 'semestral'],
        'annual': ['annual', 'yearly', 'per year'],
        'cash': ['cash', 'full payment', 'one time']
    }

    # Extract fee type
    for ftype, indicators in fee_types.items():
        if any(indicator in query_lower for indicator in indicators):
            fee_info['fee_type'] = ftype
            break

    # Extract payment term
    for pterm, indicators in payment_terms.items():
        if any(indicator in query_lower for indicator in indicators):
            fee_info['payment_term'] = pterm
            break

    # Detect program level
    graduate_indicators = ['graduate', 'master', 'phd', 'doctorate', 'masters', 'doctoral']
    if any(indicator in query_lower for indicator in graduate_indicators):
        fee_info['program_level'] = 'graduate'

    # ENHANCED: Use the same sophisticated JSON-based program extraction as programs
    # This provides better accuracy with normalization, context awareness, and priority handling
    program_info = self._extract_program_info(query)  # Uses JSON config!
    if program_info.get('program_name'):
        fee_info['program_name'] = program_info['program_name']
        fee_info['query_category'] = 'program_specific'
        print(f"📚 Enhanced JSON-based program extraction for fees: '{program_info['program_name']}'")
    else:
        # Fallback to the old method if JSON-based extraction fails
        program_keywords = self._get_comprehensive_program_keywords()
        sorted_keywords = sorted(program_keywords, key=len, reverse=True)

        for program in sorted_keywords:
            if program in query_lower:
                fee_info['program_name'] = program
                fee_info['query_category'] = 'program_specific'
                print(f"📚 Fallback program extraction for fees: '{program}'")
                break

    return fee_info

def _get_comprehensive_program_keywords(self) -> List[str]:
    """Get comprehensive list of program keywords including abbreviations"""
    # This method also uses the JSON config as a fallback source
    config = self._load_normalization_config()
    keywords = set()

    # Extract keywords from JSON config
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

                # Add description keywords
                description = abbrev_data.get('description', '')
                if description:
                    # Extract meaningful words from description
                    desc_words = [word.lower() for word in description.split()
                                if len(word) > 3 and word.lower() not in ['bachelor', 'science', 'arts', 'major']]
                    keywords.update(desc_words)

    return list(keywords)
```

#### Key Features of JSON-Based Fees Extraction

1. **Primary Method**: Uses `self._extract_program_info(query)` which loads from JSON config
2. **Same Priority System**: Applies the same "safe", "context_aware", "context_required" logic
3. **Fallback Method**: If JSON extraction fails, falls back to comprehensive keyword matching
4. **Enhanced Accuracy**: Prevents false matches like "it" being interpreted as "Information Technology"
5. **Consistent Behavior**: Fees and Programs topics use identical program detection logic

#### Benefits of JSON-Based Approach for Fees

- **Consistency**: Same program detection logic across topics
- **Accuracy**: Sophisticated context awareness prevents false matches
- **Maintainability**: Program updates only need to be made in one JSON file
- **Comprehensive Coverage**: All 59 ADDU programs with proper metadata
- **Smart Context Handling**: Handles ambiguous abbreviations intelligently

## Dialogue History & Context Management

The system maintains conversation continuity through sophisticated dialogue history tracking.

### Dialogue History Implementation

```python
class FastHybridChatbotTogether:
    def __init__(self):
        self.dialogue_history = []  # List of conversation turns
        self.session_state = {}     # Session-specific state

    def add_to_history(self, query: str, response: str):
        """Add conversation turn to dialogue history"""
        turn = {
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'turn_id': len(self.dialogue_history) + 1
        }
        self.dialogue_history.append(turn)

        # Keep only last 10 turns to manage memory
        if len(self.dialogue_history) > 10:
            self.dialogue_history = self.dialogue_history[-10:]

    def build_smart_history_context(self, current_query: str, max_tokens: int = 500) -> str:
        """Build intelligent history context for conversation continuity"""
        if not self.dialogue_history:
            return ""

        # Analyze current query for context needs
        needs_history = self._query_needs_history(current_query)
        if not needs_history:
            return ""

        # Build context from recent relevant turns
        history_parts = []
        token_count = 0

        # Start with most recent turns
        for turn in reversed(self.dialogue_history[-5:]):  # Last 5 turns
            turn_text = f"Previous: {turn['query']} → {turn['response'][:200]}..."
            turn_tokens = len(turn_text.split())

            if token_count + turn_tokens > max_tokens:
                break

            history_parts.insert(0, turn_text)
            token_count += turn_tokens

        if history_parts:
            return f"\n<|history|>\n" + "\n".join(history_parts) + "\n</|history|>\n"

        return ""

    def _query_needs_history(self, query: str) -> bool:
        """Determine if query needs conversation history"""
        query_lower = query.lower()

        # Pronouns that indicate reference to previous context
        pronouns = ['it', 'its', 'this', 'that', 'they', 'them', 'their']

        # Follow-up indicators
        followup_indicators = [
            'what about', 'how about', 'and', 'also',
            'first year', 'second year', 'third year', 'fourth year',
            '1st year', '2nd year', '3rd year', '4th year'
        ]

        # Check for pronouns or follow-up patterns
        has_pronouns = any(pronoun in query_lower.split() for pronoun in pronouns)
        has_followup = any(indicator in query_lower for indicator in followup_indicators)

        return has_pronouns or has_followup
```

### Session State Management

```python
def update_session_state(self, key: str, value: any):
    """Update session state for context preservation"""
    self.session_state[key] = value
    print(f"📝 Updated session state: {key} = {value}")

def get_session_context(self) -> Dict:
    """Get current session context"""
    return {
        'current_program': self.session_state.get('current_program'),
        'current_year': self.session_state.get('current_year'),
        'last_topic': self.session_state.get('last_topic'),
        'conversation_turns': len(self.dialogue_history)
    }
```

## Hybrid Retrieval Implementation

The system combines three retrieval methods for optimal document matching.

### TF-IDF Implementation

```python
def _load_tfidf_vectorizer(self):
    """Load or create TF-IDF vectorizer"""
    tfidf_path = os.path.join(self.embeddings_dir, "tfidf_vectorizer.pkl")

    if os.path.exists(tfidf_path):
        with open(tfidf_path, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        print(f"✅ Loaded TF-IDF vectorizer from {tfidf_path}")
    else:
        # Create new vectorizer with optimized parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,      # Vocabulary size
            stop_words='english',    # Remove common words
            ngram_range=(1, 2),      # Unigrams and bigrams
            min_df=2,                # Minimum document frequency
            max_df=0.8,              # Maximum document frequency
            sublinear_tf=True,       # Apply sublinear TF scaling
            norm='l2'                # L2 normalization
        )
        print("⚠️ Created new TF-IDF vectorizer")

def _calculate_tfidf_similarity(self, query: str, documents: List[str]) -> np.ndarray:
    """Calculate TF-IDF similarity scores"""
    if not self.tfidf_vectorizer or not documents:
        return np.zeros(len(documents))

    try:
        # Transform query and documents
        query_vector = self.tfidf_vectorizer.transform([query])
        doc_vectors = self.tfidf_vectorizer.transform(documents)

        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()

        # Apply score boosting for exact matches
        boosted_similarities = self._boost_exact_matches(query, documents, similarities)

        return boosted_similarities

    except Exception as e:
        print(f"⚠️ TF-IDF similarity calculation failed: {e}")
        return np.zeros(len(documents))

def _boost_exact_matches(self, query: str, documents: List[str], similarities: np.ndarray) -> np.ndarray:
    """Boost scores for documents with exact keyword matches"""
    query_tokens = set(query.lower().split())
    boosted_similarities = similarities.copy()

    for i, doc in enumerate(documents):
        doc_tokens = set(doc.lower().split())

        # Calculate exact match ratio
        exact_matches = len(query_tokens.intersection(doc_tokens))
        match_ratio = exact_matches / len(query_tokens) if query_tokens else 0

        # Apply boost (up to 20% increase)
        boost_factor = 1 + (match_ratio * 0.2)
        boosted_similarities[i] *= boost_factor

    return boosted_similarities
```

### Word2Vec Implementation

```python
def _load_word2vec_model(self):
    """Load Word2Vec model for semantic similarity"""
    model_path = os.path.join(self.embeddings_dir, "word2vec_model.bin")

    try:
        if os.path.exists(model_path):
            self.word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
            print(f"✅ Loaded Word2Vec model from {model_path}")
        else:
            # Try loading Google News vectors
            try:
                self.word2vec_model = api.load("word2vec-google-news-300")
                print("✅ Loaded Google News Word2Vec model")
            except:
                print("⚠️ No Word2Vec model available")
                self.word2vec_model = None
    except Exception as e:
        print(f"⚠️ Word2Vec model loading failed: {e}")
        self.word2vec_model = None

def compute_word2vec_vector(self, tokens: List[str], dim: int = 300) -> np.ndarray:
    """Compute Word2Vec vector for tokens"""
    if not tokens or not self.word2vec_model:
        return np.zeros(dim)

    vectors = []
    for token in tokens:
        try:
            if token in self.word2vec_model:
                vectors.append(self.word2vec_model[token])
        except:
            continue

    if vectors:
        # Use mean pooling for sentence representation
        return np.mean(vectors, axis=0)

    return np.zeros(dim)

def _calculate_word2vec_similarity(self, query: str, documents: List[str]) -> np.ndarray:
    """Calculate Word2Vec semantic similarity"""
    if not self.word2vec_model or not documents:
        return np.zeros(len(documents))

    # Preprocess query
    query_tokens = self._preprocess_text(query)
    query_vector = self.compute_word2vec_vector(query_tokens)

    similarities = []
    for doc in documents:
        doc_tokens = self._preprocess_text(doc)
        doc_vector = self.compute_word2vec_vector(doc_tokens)

        # Calculate cosine similarity
        if np.any(query_vector) and np.any(doc_vector):
            similarity = cosine_similarity([query_vector], [doc_vector])[0][0]
        else:
            similarity = 0.0

        similarities.append(similarity)

    return np.array(similarities)

def _preprocess_text(self, text: str) -> List[str]:
    """Preprocess text for Word2Vec"""
    # Convert to lowercase and tokenize
    tokens = text.lower().split()

    # Remove punctuation and short tokens
    tokens = [token.strip('.,!?;:"()[]{}') for token in tokens]
    tokens = [token for token in tokens if len(token) > 2]

    # Apply program name normalization
    tokens = self._normalize_program_tokens(tokens)

    return tokens
```

### Hybrid Scoring Algorithm

```python
def retrieve_documents_by_topic_hybrid(self, query: str, topic_id: str, top_k: int = 2) -> List[Dict]:
    """
    HYBRID TOPIC + SEMANTIC RETRIEVAL:
    1. Filter documents by topic keywords
    2. Generate TF-IDF + Word2Vec vectors for query and filtered documents
    3. Calculate semantic similarity using cosine similarity
    4. Combine topic relevance (60%) + semantic similarity (40%)
    """
    print(f"🔬 Hybrid topic + semantic retrieval for: '{query}' (topic: {topic_id})")

    # Normalize query
    normalized_query = self._normalize_program_acronyms(query)
    normalized_query = self._normalize_school_abbreviations(normalized_query)

    try:
        collection = ChromaService.get_client().get_or_create_collection(name=self.chroma_collection_name)

        # Get topic keywords for filtering
        topic_keywords = get_topic_keywords(topic_id)
        if not topic_keywords:
            print(f"⚠️ No keywords found for topic '{topic_id}', using generic retrieval")
            return self.retrieve_documents_by_topic_keywords_simple(query, topic_id, top_k)

        # Build keyword filter for ChromaDB
        keyword_conditions = []
        for keyword in topic_keywords:
            keyword_conditions.append({"keywords": {"$regex": f".*\\b{keyword}\\b.*"}})

        where_clause = {
            "$and": [
                {"source": "pdf_scrape"},
                {"$or": keyword_conditions}
            ]
        }

        # Get filtered documents
        all_docs = collection.get(
            where=where_clause,
            include=["documents", "metadatas"]
        )

        all_ids = all_docs.get('ids', [])
        all_contents = all_docs.get('documents', [])
        all_metadatas = all_docs.get('metadatas', [])

        if not all_contents:
            print(f"⚠️ No documents found for topic '{topic_id}'")
            return []

        print(f"📚 Found {len(all_contents)} topic-filtered documents")

        # Calculate hybrid scores
        tfidf_scores = self._calculate_tfidf_similarity(normalized_query, all_contents)
        word2vec_scores = self._calculate_word2vec_similarity(normalized_query, all_contents)

        # Combine scores with weighted average
        # TF-IDF: 60% (keyword matching)
        # Word2Vec: 40% (semantic similarity)
        hybrid_scores = 0.6 * tfidf_scores + 0.4 * word2vec_scores

        # Apply topic-specific boosting
        boosted_scores = self._apply_topic_boosting(hybrid_scores, all_metadatas, topic_id, query)

        # Create scored results
        scored_results = []
        for i, (doc_id, content, metadata) in enumerate(zip(all_ids, all_contents, all_metadatas)):
            scored_results.append({
                'id': doc_id,
                'content': content,
                'metadata': metadata,
                'relevance': float(boosted_scores[i]),
                'tfidf_score': float(tfidf_scores[i]),
                'word2vec_score': float(word2vec_scores[i]),
                'hybrid_score': float(hybrid_scores[i])
            })

        # Sort by final boosted score and return top_k
        scored_results.sort(key=lambda x: x['relevance'], reverse=True)

        # Log retrieval strategy
        self._log_retrieval_strategy(query, topic_id, len(scored_results), "hybrid")

        return scored_results[:top_k]

    except Exception as e:
        print(f"❌ Hybrid retrieval error: {e}")
        return self.retrieve_documents_by_topic_keywords_simple(query, topic_id, top_k)

def _apply_topic_boosting(self, scores: np.ndarray, metadatas: List[Dict],
                         topic_id: str, query: str) -> np.ndarray:
    """Apply topic-specific score boosting"""
    boosted_scores = scores.copy()

    for i, metadata in enumerate(metadatas):
        filename = metadata.get('filename', '').lower()
        doc_type = metadata.get('document_type', '')
        keywords = metadata.get('keywords', '').lower()

        # Topic-specific boosting rules
        if topic_id == 'admissions_enrollment':
            if doc_type in ['admission', 'enrollment', 'scholarship']:
                boosted_scores[i] *= 1.2  # 20% boost
            if 'requirement' in filename or 'admission' in filename:
                boosted_scores[i] *= 1.1  # 10% boost

        elif topic_id == 'programs_courses':
            if doc_type in ['academic', 'curriculum']:
                boosted_scores[i] *= 1.2
            if 'curriculum' in filename or 'program' in filename:
                boosted_scores[i] *= 1.1
            # Boost for program-specific matches
            program_info = self._extract_program_info(query)
            if program_info['program_name'] and program_info['program_name'] in keywords:
                boosted_scores[i] *= 1.3  # 30% boost for program match

        elif topic_id == 'fees':
            if doc_type in ['fees', 'payment']:
                boosted_scores[i] *= 1.2
            if 'fee' in filename or 'tuition' in filename:
                boosted_scores[i] *= 1.1

    return boosted_scores
```

## Response Generation & Prompt Engineering

### Topic-Specific Prompt Templates

The system uses specialized prompts for each topic to ensure accurate and relevant responses.

#### Admissions & Enrollment Prompts

```python
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
```

#### Programs & Courses Prompts

```python
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

=== CURRICULUM QUERIES ===
- **YEAR-BASED DISPLAY**: Curriculum queries show subjects for the requested year level
- **DEFAULT YEAR**: If no year is specified, default to Year 1 curriculum
- **FOLLOW-UP SUPPORT**: Users can ask "what about 2nd year" or "show me 3rd year" to navigate through different years
- **MULTI-PROGRAM QUERIES**: If multiple programs are mentioned (e.g., "compare CS and IT curriculum"), format as a comparison showing both curricula side-by-side with clear separation between programs
- **FORMAT**:
  * Show program name/acronym
  * Show year level
  * List subjects organized by semester (First Semester, Second Semester, Summer if applicable)
  * **Semester headers**: Format semester names in bold with total Credit Units (e.g., **First Semester (21.0 CU)**, **Second Semester (21.0 CU)**)
  * **Empty semesters**: If a semester has no courses, DO NOT show that semester header at all - omit it completely
  * Include course codes, titles, and credits for each course
  * Include curriculum PDF link if available in the document
- **SESSION CONTINUITY**: Remember which program and year was last displayed for follow-up queries

=== LINK HANDLING ===
- **ALLOW LINKS**: Include links if they are explicitly present in the source document content
- **NO FABRICATION**: NEVER create, invent, or fabricate URLs
- **HYPERLINKS**: If URLs exist, format them as clickable hyperlinks using Markdown syntax: [link text](URL)
- **CRITICAL**: Use the EXACT URL from the document content - DO NOT modify, autocorrect, or change ANY part of the URL
- **NO CORRECTIONS**: Do NOT fix typos in URLs, do NOT change "Technolgy" to "Technology", do NOT modify any part of the original URL
- **PRESERVE ORIGINAL**: Copy the URL character-for-character exactly as it appears in the source document
- **FORMAT**: If URLs exist, use "For more information about [topic], head to this link: [link text](EXACT_URL_FROM_DOCUMENT)"
- **NO LINKS RULE**: If no URLs are present in the source documents, do NOT mention links at all"""
```

#### Fees & Payments Prompts

```python
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
```

### Main Response Generation Flow

```python
def _process_topic_query(self, query: str, topic_id: str) -> Dict:
    """Process query within guided conversation context"""

    # 1. Query Enhancement and Normalization
    enhanced_query = self._enhance_query_with_context(query, topic_id)
    normalized_query = self._normalize_program_acronyms(enhanced_query)
    normalized_query = self._normalize_school_abbreviations(normalized_query)

    print(f"🔍 Processing topic query: '{query}' → '{normalized_query}' (topic: {topic_id})")

    # 2. Intent Classification
    intent_info = self._classify_intent(normalized_query, topic_id)
    print(f"🎯 Classified intent: {intent_info}")

    # 3. Session State Updates
    if topic_id == 'programs_courses':
        program_info = self._extract_program_info(normalized_query)
        if program_info['program_name']:
            self.update_session_state('current_program', program_info['program_name'])
        if program_info['year_level']:
            self.update_session_state('current_year', program_info['year_level'])

    elif topic_id == 'fees':
        fee_info = self._extract_fee_info(normalized_query)
        if fee_info['program_name']:
            self.update_session_state('current_program', fee_info['program_name'])

    # 4. Dynamic Retrieval Strategy Selection
    dynamic_top_k = self._calculate_dynamic_top_k(intent_info, topic_id)
    print(f"🎯 Using top_k={dynamic_top_k} for retrieval")

    # 5. Specialized Document Retrieval
    relevant_docs = self.retrieve_documents_by_topic_specialized(enhanced_query, topic_id, top_k=dynamic_top_k)

    if not relevant_docs:
        return {
            'response': "I apologize, but I couldn't find relevant information for your query. Please try rephrasing your question or contact the admissions office for assistance.",
            'sources': [],
            'retrieval_time': 0,
            'topic_id': topic_id
        }

    # 6. Context Building
    doc_context = "\n\n".join([
        f"Source: {doc.get('id','')}\n{doc['content']}"
        for doc in relevant_docs[:dynamic_top_k]
    ])

    # 7. Dialogue History Integration
    history_context = ""
    if self.dialogue_history:
        # Calculate available token budget for history
        base_prompt_estimate = f"System instructions + Context: {doc_context} + Query: {enhanced_query}"
        base_tokens = len(base_prompt_estimate.split())
        available_for_history = 3500 - base_tokens  # Conservative token limit

        # Use smart history building
        history_context = self.build_smart_history_context(query, available_for_history)
        print(f"📜 Built history context: {len(history_context)} chars (~{len(history_context.split())} tokens)")

    # 8. Topic-Specific Prompt Construction
    topic_info = get_topic_info(topic_id)
    topic_label = topic_info.get('label', topic_id) if topic_info else topic_id
    topic_specific_instructions = self._get_topic_specific_instructions(topic_id)

    prompt = f"""<|system|>
You are an ADDU (Ateneo de Davao University) Admissions Assistant. You provide accurate, helpful information based strictly on the provided context documents.

CRITICAL URL RULE:
- NEVER create, invent, fabricate, or hallucinate URLs
- If no URLs are present in the source documents, do NOT mention links, URLs, or any reference to external resources
- Use topic-specific link handling rules (see topic instructions below)

GENERAL RESPONSE RULES:
- Be direct and concise
- Use simple, clear formatting
- No introductory phrases like "Based on the provided documentation"
- No closing phrases like "I hope this helps"
- Start directly with the answer
- Use numbered lists for steps/processes
- Use bullet points for items/lists
- Bold important terms only when necessary
- Maintain conversation context and continuity

CURRENT TOPIC: {topic_label}

{topic_specific_instructions}
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

    # 9. LLM Generation with Streaming
    start_time = time.time()
    response = stream_response(prompt, max_tokens=1024)
    generation_time = time.time() - start_time

    # 10. Update Dialogue History
    self.add_to_history(query, response)

    # 11. Log Performance Metrics
    self._log_query_performance({
        'query': query,
        'topic_id': topic_id,
        'intent': intent_info,
        'docs_retrieved': len(relevant_docs),
        'generation_time': generation_time,
        'response_length': len(response)
    })

    return {
        'response': response,
        'sources': relevant_docs,
        'retrieval_time': generation_time,
        'topic_id': topic_id,
        'intent_info': intent_info,
        'session_context': self.get_session_context()
    }
```

## Admin System & Document Processing

### Document Upload and Processing Pipeline

```python
@csrf_exempt
def admin_upload_view(request):
    """Handle document upload with comprehensive metadata processing"""
    if request.method == "POST":
        try:
            # 1. Extract upload parameters
            file = request.FILES.get('file')
            folder_id = request.POST.get('folder_id')
            document_type = request.POST.get('document_type', 'other')
            keywords = request.POST.get('keywords', '')

            # 2. Validate file type and size
            allowed_extensions = ['.pdf', '.txt', '.doc', '.docx', '.csv', '.xlsx', '.xls']
            file_ext = os.path.splitext(file.name)[1].lower()

            if file_ext not in allowed_extensions:
                return JsonResponse({
                    "error": f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
                }, status=400)

            # Check file size (max 50MB)
            if file.size > 50 * 1024 * 1024:
                return JsonResponse({
                    "error": "File size exceeds 50MB limit"
                }, status=400)

            # 3. Upload to Supabase Storage
            supabase = get_supabase_client()
            file_bytes = file.read()

            # Generate unique filename to prevent conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{file.name}"

            upload_result = supabase.storage.from_("documents").upload(
                f"uploads/{unique_filename}",
                file_bytes,
                file_options={
                    "content-type": file.content_type,
                    "cache-control": "3600"
                }
            )

            # 4. Create database record with metadata
            folder = DocumentFolder.objects.get(id=folder_id)
            doc_metadata = DocumentMetadata.objects.create(
                filename=file.name,
                original_filename=file.name,
                stored_filename=unique_filename,
                folder=folder,
                document_type=document_type,
                keywords=keywords,
                file_size=len(file_bytes),
                file_extension=file_ext,
                content_type=file.content_type,
                supabase_path=upload_result.path,
                upload_timestamp=datetime.now(),
                synced_to_chroma=False
            )

            # 5. Process and embed document
            processing_result = sync_supabase_to_chroma_improved()

            # 6. Update search indices
            update_search_indices()

            return JsonResponse({
                "message": f"Document '{file.name}' uploaded and processed successfully",
                "document_id": doc_metadata.id,
                "processing_result": processing_result,
                "file_info": {
                    "size": len(file_bytes),
                    "type": file.content_type,
                    "extension": file_ext
                }
            })

        except DocumentFolder.DoesNotExist:
            return JsonResponse({"error": "Invalid folder ID"}, status=400)
        except Exception as e:
            print(f"Upload error: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Method not allowed"}, status=405)
```

### Document Processing and Embedding

```python
def sync_supabase_to_chroma_improved():
    """Sync documents from Supabase to ChromaDB with improved processing"""

    try:
        # Initialize services
        supabase = get_supabase_client()
        chroma_service = ChromaService()
        collection = chroma_service.get_client().get_or_create_collection(name="documents")

        # Get unsynced documents
        unsynced_docs = DocumentMetadata.objects.filter(synced_to_chroma=False)
        processed_count = 0
        failed_count = 0

        for doc in unsynced_docs:
            try:
                print(f"📄 Processing document: {doc.filename}")

                # 1. Download from Supabase
                file_data = supabase.storage.from_("documents").download(doc.supabase_path)

                # 2. Extract text based on file type
                text_content = extract_text_by_type(file_data, doc.filename, doc.file_extension)

                if not text_content or len(text_content.strip()) < 50:
                    print(f"⚠️ Insufficient text content in {doc.filename}")
                    continue

                # 3. Clean and preprocess text
                cleaned_content = clean_and_preprocess_text(text_content)

                # 4. Generate embeddings
                embedding = embed_text(cleaned_content)

                # 5. Prepare comprehensive metadata
                metadata = {
                    'filename': doc.filename,
                    'original_filename': doc.original_filename,
                    'stored_filename': doc.stored_filename,
                    'folder_name': doc.folder.name,
                    'folder_path': doc.folder.folder_path,
                    'document_type': doc.document_type,
                    'keywords': doc.keywords,
                    'file_size': doc.file_size,
                    'file_extension': doc.file_extension,
                    'content_type': doc.content_type,
                    'source': 'pdf_scrape',
                    'created_at': doc.created_at.isoformat(),
                    'upload_timestamp': doc.upload_timestamp.isoformat(),
                    'text_length': len(cleaned_content),
                    'word_count': len(cleaned_content.split()),
                    'processing_version': '2.0'
                }

                # 6. Add to ChromaDB
                collection.add(
                    embeddings=[embedding],
                    documents=[cleaned_content],
                    metadatas=[metadata],
                    ids=[f"doc_{doc.id}"]
                )

                # 7. Mark as synced and update processing info
                doc.synced_to_chroma = True
                doc.text_content_length = len(cleaned_content)
                doc.word_count = len(cleaned_content.split())
                doc.processing_timestamp = datetime.now()
                doc.save()

                processed_count += 1
                print(f"✅ Successfully processed: {doc.filename}")

            except Exception as e:
                failed_count += 1
                print(f"❌ Failed to process {doc.filename}: {e}")

                # Log processing error
                doc.processing_error = str(e)
                doc.processing_timestamp = datetime.now()
                doc.save()
                continue

        # Update TF-IDF vectorizer with new documents
        if processed_count > 0:
            update_tfidf_vectorizer()

        return {
            "status": "success",
            "processed_count": processed_count,
            "failed_count": failed_count,
            "total_documents": DocumentMetadata.objects.filter(synced_to_chroma=True).count()
        }

    except Exception as e:
        print(f"❌ Sync process failed: {e}")
        return {"status": "error", "message": str(e)}

def extract_text_by_type(file_data: bytes, filename: str, file_extension: str) -> str:
    """Extract text content based on file type"""
    try:
        if file_extension.lower() == '.pdf':
            return extract_text_from_pdf(file_data)
        elif file_extension.lower() in ['.xlsx', '.xls', '.csv']:
            return extract_text_from_excel_csv(file_data, filename)
        elif file_extension.lower() in ['.doc', '.docx']:
            return extract_text_from_word(file_data)
        elif file_extension.lower() == '.txt':
            return file_data.decode('utf-8', errors='ignore')
        else:
            # Try to decode as text
            return file_data.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"❌ Text extraction failed for {filename}: {e}")
        return ""

def clean_and_preprocess_text(text: str) -> str:
    """Clean and preprocess extracted text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)

    # Remove very short lines (likely artifacts)
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]

    # Rejoin and normalize spacing
    cleaned_text = '\n'.join(cleaned_lines)
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # Normalize paragraph breaks

    return cleaned_text.strip()
```

## Database Schema & Models

### Core Models

```python
# Topic Management
class Topic(models.Model):
    """Topics for guided conversation"""
    topic_id = models.CharField(max_length=50, unique=True)
    label = models.CharField(max_length=255)
    description = models.TextField()
    retrieval_strategy = models.CharField(max_length=50, default='generic')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'chatbot_topic'

    def __str__(self):
        return f"{self.label} ({self.topic_id})"

class TopicKeyword(models.Model):
    """Keywords associated with topics for document filtering"""
    topic = models.ForeignKey(Topic, on_delete=models.CASCADE, related_name='keywords')
    keyword = models.CharField(max_length=255)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.CharField(max_length=255, blank=True)

    class Meta:
        db_table = 'chatbot_topickeyword'
        unique_together = ['topic', 'keyword']

    def __str__(self):
        return f"{self.topic.topic_id}: {self.keyword}"

# Document Management
class DocumentFolder(models.Model):
    """Hierarchical folder structure for document organization"""
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    color = models.CharField(max_length=7, default='#063970')  # Hex color
    parent_folder = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'chatbot_documentfolder'

    @property
    def folder_path(self):
        """Get full folder path (e.g., 'Parent / Child / Grandchild')"""
        if self.parent_folder:
            return f"{self.parent_folder.folder_path} / {self.name}"
        return self.name

    @property
    def depth_level(self):
        """Get folder depth level"""
        level = 0
        current = self.parent_folder
        while current:
            level += 1
            current = current.parent_folder
        return level

    def get_all_children(self):
        """Get all child folders recursively"""
        children = list(self.documentfolder_set.all())
        for child in list(children):
            children.extend(child.get_all_children())
        return children

    def __str__(self):
        return self.folder_path

class DocumentMetadata(models.Model):
    """Comprehensive document metadata and storage information"""
    DOCUMENT_TYPE_CHOICES = [
        ('admission', 'Admission Requirements'),
        ('enrollment', 'Enrollment Process'),
        ('scholarship', 'Scholarships & Financial Aid'),
        ('academic', 'Academic Programs'),
        ('curriculum', 'Curriculum & Courses'),
        ('fees', 'Fees & Payments'),
        ('policy', 'Policies & Procedures'),
        ('contact', 'Contact Information'),
        ('other', 'Other'),
    ]

    # File Information
    filename = models.CharField(max_length=255)
    original_filename = models.CharField(max_length=255)
    stored_filename = models.CharField(max_length=255)
    file_extension = models.CharField(max_length=10)
    content_type = models.CharField(max_length=100)
    file_size = models.IntegerField()

    # Organization
    folder = models.ForeignKey(DocumentFolder, on_delete=models.CASCADE)
    document_type = models.CharField(max_length=20, choices=DOCUMENT_TYPE_CHOICES)
    keywords = models.TextField(help_text="Comma-separated keywords")

    # Storage
    supabase_path = models.CharField(max_length=500)

    # Processing Status
    synced_to_chroma = models.BooleanField(default=False)
    text_content_length = models.IntegerField(null=True, blank=True)
    word_count = models.IntegerField(null=True, blank=True)
    processing_error = models.TextField(blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    upload_timestamp = models.DateTimeField()
    processing_timestamp = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'chatbot_documentmetadata'
        indexes = [
            models.Index(fields=['document_type']),
            models.Index(fields=['synced_to_chroma']),
            models.Index(fields=['folder']),
        ]

    @property
    def file_size_mb(self):
        """Get file size in MB"""
        return round(self.file_size / (1024 * 1024), 2)

    @property
    def is_processed(self):
        """Check if document has been processed successfully"""
        return self.synced_to_chroma and not self.processing_error

    def __str__(self):
        return f"{self.filename} ({self.document_type})"

# Conversation Management
class Conversation(models.Model):
    """Conversation sessions for dialogue history tracking"""
    session_id = models.CharField(max_length=255, unique=True)
    user_identifier = models.CharField(max_length=255, blank=True)
    current_topic = models.CharField(max_length=50, blank=True)
    conversation_state = models.CharField(max_length=50, default='topic_selection')
    session_data = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_activity = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'chatbot_conversation'

    def __str__(self):
        return f"Conversation {self.session_id}"

class ConversationTurn(models.Model):
    """Individual turns in a conversation"""
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='turns')
    turn_number = models.IntegerField()
    user_input = models.TextField()
    bot_response = models.TextField()
    topic_id = models.CharField(max_length=50, blank=True)
    intent_classification = models.JSONField(default=dict)
    retrieved_documents = models.JSONField(default=list)
    response_time = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'chatbot_conversationturn'
        unique_together = ['conversation', 'turn_number']
        indexes = [
            models.Index(fields=['conversation', 'turn_number']),
            models.Index(fields=['topic_id']),
        ]

    def __str__(self):
        return f"Turn {self.turn_number} in {self.conversation.session_id}"
```

## API Endpoints & Request Flow

### Guided Chat Endpoint

```python
@csrf_exempt
def guided_chat_view(request):
    """Main guided chat endpoint with comprehensive request handling"""
    if request.method == 'POST':
        try:
            # 1. Parse request data
            data = json.loads(request.body)
            user_input = data.get('user_input', '')
            action_type = data.get('action_type', 'message')
            action_data = data.get('action_data')
            session_id = data.get('session_id')

            # 2. Initialize or retrieve conversation
            if not session_id:
                session_id = str(uuid.uuid4())

            conversation, created = Conversation.objects.get_or_create(
                session_id=session_id,
                defaults={
                    'conversation_state': 'topic_selection',
                    'session_data': {}
                }
            )

            # 3. Initialize chatbot with conversation context
            chatbot = FastHybridChatbotTogether()
            chatbot.load_conversation_history(conversation)

            # 4. Process guided conversation
            start_time = time.time()
            result = chatbot.process_guided_conversation(
                user_input=user_input,
                action_type=action_type,
                action_data=action_data
            )
            processing_time = time.time() - start_time

            # 5. Update conversation state
            conversation.current_topic = result.get('current_topic')
            conversation.conversation_state = result.get('conversation_state')
            conversation.session_data.update(result.get('session_data', {}))
            conversation.last_activity = datetime.now()
            conversation.save()

            # 6. Save conversation turn
            if result.get('response') and user_input:
                turn = ConversationTurn.objects.create(
                    conversation=conversation,
                    turn_number=conversation.turns.count() + 1,
                    user_input=user_input,
                    bot_response=result.get('response', ''),
                    topic_id=result.get('current_topic', ''),
                    intent_classification=result.get('intent_info', {}),
                    retrieved_documents=result.get('sources', []),
                    response_time=processing_time
                )

            # 7. Prepare response
            response_data = {
                'response': result.get('response'),
                'conversation_state': result.get('conversation_state'),
                'current_topic': result.get('current_topic'),
                'session_id': session_id,
                'buttons': result.get('buttons', []),
                'sources': result.get('sources', []),
                'processing_time': processing_time,
                'turn_number': conversation.turns.count(),
                '_debug': {
                    'intent_info': result.get('intent_info'),
                    'session_context': result.get('session_context'),
                    'retrieval_strategy': result.get('retrieval_strategy')
                }
            }

            return JsonResponse(response_data)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            print(f"❌ Guided chat error: {e}")
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Method not allowed'}, status=405)
```

### Topic Management Endpoints

```python
@csrf_exempt
def topics_view(request):
    """Get available topics for guided conversation"""
    if request.method == 'GET':
        try:
            topics = Topic.objects.filter(is_active=True).order_by('label')
            topics_data = []

            for topic in topics:
                topic_data = {
                    'id': topic.topic_id,
                    'label': topic.label,
                    'description': topic.description,
                    'keywords': [kw.keyword for kw in topic.keywords.filter(is_active=True)],
                    'retrieval_strategy': topic.retrieval_strategy,
                    'document_count': get_topic_document_count(topic.topic_id)
                }
                topics_data.append(topic_data)

            return JsonResponse({
                'topics': topics_data,
                'total_count': len(topics_data)
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Method not allowed'}, status=405)

def get_topic_document_count(topic_id: str) -> int:
    """Get count of documents associated with a topic"""
    try:
        topic_keywords = get_topic_keywords(topic_id)
        if not topic_keywords:
            return 0

        # Count documents that match topic keywords
        keyword_conditions = Q()
        for keyword in topic_keywords:
            keyword_conditions |= Q(keywords__icontains=keyword)

        count = DocumentMetadata.objects.filter(
            keyword_conditions,
            synced_to_chroma=True
        ).count()

        return count
    except:
        return 0
```

## Performance Optimization

### Caching Strategy

```python
from django.core.cache import cache
from django.conf import settings

class CacheManager:
    """Centralized cache management for chatbot operations"""

    CACHE_TIMEOUTS = {
        'topic_keywords': 3600,      # 1 hour
        'document_embeddings': 7200,  # 2 hours
        'tfidf_vectors': 3600,       # 1 hour
        'conversation_context': 1800, # 30 minutes
        'retrieval_results': 600     # 10 minutes
    }

    @staticmethod
    def get_topic_keywords(topic_id: str) -> List[str]:
        """Get topic keywords with caching"""
        cache_key = f"topic_keywords_{topic_id}"
        keywords = cache.get(cache_key)

        if keywords is None:
            try:
                topic = Topic.objects.get(topic_id=topic_id, is_active=True)
                keywords = list(topic.keywords.filter(is_active=True).values_list('keyword', flat=True))
                cache.set(cache_key, keywords, CacheManager.CACHE_TIMEOUTS['topic_keywords'])
            except Topic.DoesNotExist:
                keywords = []
                cache.set(cache_key, keywords, 300)  # Short cache for missing topics

        return keywords

    @staticmethod
    def cache_retrieval_results(query: str, topic_id: str, results: List[Dict]):
        """Cache retrieval results for similar queries"""
        cache_key = f"retrieval_{hashlib.md5(f'{query}_{topic_id}'.encode()).hexdigest()}"
        cache.set(cache_key, results, CacheManager.CACHE_TIMEOUTS['retrieval_results'])

    @staticmethod
    def get_cached_retrieval_results(query: str, topic_id: str) -> Optional[List[Dict]]:
        """Get cached retrieval results"""
        cache_key = f"retrieval_{hashlib.md5(f'{query}_{topic_id}'.encode()).hexdigest()}"
        return cache.get(cache_key)

    @staticmethod
    def invalidate_topic_cache(topic_id: str):
        """Invalidate all caches related to a topic"""
        cache_keys = [
            f"topic_keywords_{topic_id}",
            f"topic_documents_{topic_id}",
            f"topic_config_{topic_id}"
        ]
        cache.delete_many(cache_keys)
```

### Database Query Optimization

```python
class OptimizedQueryManager:
    """Optimized database queries for chatbot operations"""

    @staticmethod
    def get_documents_by_topic_optimized(topic_id: str) -> QuerySet:
        """Get documents for a topic with optimized queries"""
        topic_keywords = CacheManager.get_topic_keywords(topic_id)

        if not topic_keywords:
            return DocumentMetadata.objects.none()

        # Build efficient OR query for keywords
        keyword_q = Q()
        for keyword in topic_keywords:
            keyword_q |= Q(keywords__icontains=keyword)

        return DocumentMetadata.objects.select_related('folder').filter(
            keyword_q,
            synced_to_chroma=True
        ).only(
            'id', 'filename', 'document_type', 'keywords',
            'folder__name', 'file_size', 'created_at'
        )

    @staticmethod
    def get_conversation_history_optimized(session_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history with optimized queries"""
        try:
            conversation = Conversation.objects.get(session_id=session_id)
            turns = conversation.turns.order_by('-turn_number')[:limit]

            return [
                {
                    'turn_number': turn.turn_number,
                    'user_input': turn.user_input,
                    'bot_response': turn.bot_response,
                    'topic_id': turn.topic_id,
                    'created_at': turn.created_at.isoformat()
                }
                for turn in reversed(turns)
            ]
        except Conversation.DoesNotExist:
            return []
```

### Memory Management

```python
class MemoryManager:
    """Memory management for large-scale operations"""

    @staticmethod
    def batch_process_documents(documents: List[DocumentMetadata], batch_size: int = 50):
        """Process documents in batches to manage memory"""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            yield batch

    @staticmethod
    def cleanup_old_conversations(days_old: int = 30):
        """Clean up old conversation data"""
        cutoff_date = datetime.now() - timedelta(days=days_old)

        # Delete old conversations and their turns
        old_conversations = Conversation.objects.filter(
            last_activity__lt=cutoff_date
        )

        deleted_count = 0
        for conversation in old_conversations:
            deleted_count += conversation.turns.count()
            conversation.delete()

        return deleted_count

    @staticmethod
    def optimize_dialogue_history(chatbot_instance, max_turns: int = 10):
        """Optimize dialogue history to prevent memory bloat"""
        if len(chatbot_instance.dialogue_history) > max_turns:
            # Keep only the most recent turns
            chatbot_instance.dialogue_history = chatbot_instance.dialogue_history[-max_turns:]

            # Optionally compress older turns
            for turn in chatbot_instance.dialogue_history[:-5]:  # Compress all but last 5
                if len(turn['response']) > 500:
                    turn['response'] = turn['response'][:500] + "..."
```

This comprehensive documentation covers all aspects of the ADDU Admissions Chatbot system, from low-level retrieval algorithms to high-level conversation management. Each section provides detailed code examples and technical explanations of how the system operates in practice.
