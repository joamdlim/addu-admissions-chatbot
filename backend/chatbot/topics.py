"""
Topic definitions for guided conversation flow with keyword-based document filtering.
Each topic has associated keywords that are matched against document metadata keywords field.
"""

# Predefined topics with keyword mappings for document filtering
# Redesigned with 3 focused topics and specialized retrieval strategies
TOPICS = {
    'admissions_enrollment': {
        'label': 'General Admissions and Enrollment',
        'keywords': [
            # General admissions
            'admission', 'requirements', 'application', 'entrance', 'apply', 'qualifying',
            # Enrollment process
            'enrollment', 'registration', 'enroll', 'register', 'sign up',
            # Student types
            'new student', 'freshman', 'first year', 'incoming',
            'scholar', 'scholarship', 'financial aid', 'grant', 'funding',
            'transferee', 'transfer', 'shifter', 'lateral entry',
            'international', 'foreign student', 'foreign', 'overseas',
            # Required documents and requirements (merged from requirements topic)
            'documents', 'documents needed', 'requirements list',
            'transcript', 'diploma', 'certificate', 'form 137', 'form 138',
            'birth certificate', 'medical certificate', 'clearance',
            'recommendation letter', 'essay', 'portfolio',
            'entrance exam', 'interview', 'assessment',
            # Process terms
            'process', 'procedure', 'steps', 'how to apply'
        ],
        'description': 'Learn about admissions, enrollment processes, requirements, and required documents for all student types (new students, scholars, transferees, international students)',
        'retrieval_strategy': 'admissions_specialized'
    },
    'programs_courses': {
        'label': 'Programs and Courses',
        'keywords': [
            # Programs and degrees
            'program', 'degree', 'course', 'major', 'bachelor',
            'undergraduate',
            'college', 'school', 'department', 'faculty',
            'BS', 'BA'
            # Curriculum and courses
            'curriculum', 'courses', 'subjects', 'syllabus', 'course outline', 'academic plan',
            'first year', 'second year', 'third year', 'fourth year',
            'semester', 'units', 'credits',
            # Specific program names (can be expanded)
            'computer science', 'information technology', 'business administration',
            'engineering', 'nursing', 'education', 'psychology'
        ],
        'description': 'Learn about available academic programs, degrees, curriculum, and course offerings',
        'retrieval_strategy': 'programs_specialized'
    },
    'fees': {
        'label': 'Fees',
        'keywords': [
            # Basic fee terms
            'fees', 'tuition', 'payment', 'cost', 'price', 'amount', 'billing',
            # Payment terms
            'payment plan', 'installment', 'due date', 'payment schedule',
            'down payment', 'balance', 'discount',
            # Specific fee types
            'miscellaneous fees', 'laboratory fees', 'library fees',
            'graduation fee', 'examination fee', 'registration fee',
            'development fee', 'student activities fee',
            # Financial terms
            'scholarship', 'financial aid', 'grant', 'subsidy'
        ],
        'description': 'Learn about tuition fees, payment options, and financial information',
        'retrieval_strategy': 'fees_specialized'
    }
}

# Conversation states
CONVERSATION_STATES = {
    'TOPIC_SELECTION': 'topic_selection',
    'TOPIC_CONVERSATION': 'topic_conversation',
    'FOLLOW_UP': 'follow_up'
}

# Button configurations for different states
BUTTON_CONFIGS = {
    'topic_selection': {
        'buttons': [
            {'id': topic_id, 'label': topic_data['label'], 'type': 'topic'}
            for topic_id, topic_data in TOPICS.items()
        ],
        'input_enabled': False,
        'message': 'Please select a topic you\'d like to learn about:'
    },
    'follow_up': {
        'buttons': [
            {'id': 'ask_another', 'label': 'Ask Another Question', 'type': 'action'},
            {'id': 'change_topic', 'label': 'Change Topic', 'type': 'action'}
        ],
        'input_enabled': False,
        'message': None  # No additional message needed
    },
    'topic_conversation': {
        'buttons': [],
        'input_enabled': True,
        'message': None
    }
}

def get_topic_keywords(topic_id):
    """Get keywords for a specific topic"""
    return TOPICS.get(topic_id, {}).get('keywords', [])

def get_all_topic_keywords():
    """Get all keywords from all topics as a flat list"""
    all_keywords = []
    for topic_data in TOPICS.values():
        all_keywords.extend(topic_data.get('keywords', []))
    return all_keywords

def find_matching_topics(query_text):
    """Find topics that match keywords in the query text"""
    query_lower = query_text.lower()
    matching_topics = []
    
    for topic_id, topic_data in TOPICS.items():
        keywords = topic_data.get('keywords', [])
        matches = sum(1 for keyword in keywords if keyword.lower() in query_lower)
        if matches > 0:
            matching_topics.append({
                'topic_id': topic_id,
                'topic_data': topic_data,
                'match_count': matches,
                'match_ratio': matches / len(keywords)
            })
    
    # Sort by match count (descending)
    matching_topics.sort(key=lambda x: x['match_count'], reverse=True)
    return matching_topics

def get_topic_info(topic_id):
    """Get complete information about a topic"""
    return TOPICS.get(topic_id)

def get_topic_retrieval_strategy(topic_id):
    """Get the specialized retrieval strategy for a topic"""
    topic_info = TOPICS.get(topic_id, {})
    return topic_info.get('retrieval_strategy', 'generic')

# Topic-specific retrieval strategy mapping
TOPIC_RETRIEVAL_STRATEGIES = {
    'admissions_specialized': {
        'description': 'Specialized retrieval for admissions, enrollment, requirements, and documents',
        'document_types': ['admission', 'enrollment', 'scholarship'],
        'metadata_priorities': {
            'filename': 0.4,  # High priority for well-named admission documents
            'keywords': 0.3,  # Medium-high for keyword matching
            'content': 0.3    # Medium for content matching
        },
        'student_type_detection': True,  # Detect new/transfer/international/scholar
        'requirement_extraction': True   # Extract specific document requirements
    },
    'programs_specialized': {
        'description': 'Specialized retrieval for programs, courses, and curriculum',
        'document_types': ['academic', 'curriculum'],
        'metadata_priorities': {
            'filename': 0.5,  # Very high - curriculum files are systematically named
            'keywords': 0.3,  # Medium-high for program matching
            'content': 0.2    # Lower for content (filenames are more reliable)
        },
        'program_extraction': True,      # Extract program names from query
        'year_level_detection': True,    # Detect 1st/2nd/3rd/4th year queries
        'course_code_matching': True     # Match course codes if present
    },
    'fees_specialized': {
        'description': 'Specialized retrieval for fees, payments, and financial information',
        'document_types': ['fees', 'financial'],
        'metadata_priorities': {
            'filename': 0.3,  # Medium - fee documents may have varied naming
            'keywords': 0.4,  # High priority for fee-related keywords
            'content': 0.3    # Medium for amount and payment term matching
        },
        'amount_extraction': True,       # Extract fee amounts from query
        'payment_term_detection': True,  # Detect payment schedule queries
        'program_level_filtering': True  # Filter by undergraduate/graduate fees
    }
}

def get_retrieval_strategy_config(strategy_name):
    """Get configuration for a specific retrieval strategy"""
    return TOPIC_RETRIEVAL_STRATEGIES.get(strategy_name, {})
