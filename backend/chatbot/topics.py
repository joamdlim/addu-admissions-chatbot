"""
Topic definitions for guided conversation flow with keyword-based document filtering.
Each topic has associated keywords that are matched against document metadata keywords field.
"""

# All topic data is now stored in the database
# This file only contains helper functions and constants

# Conversation states
CONVERSATION_STATES = {
    'TOPIC_SELECTION': 'topic_selection',
    'TOPIC_CONVERSATION': 'topic_conversation',
    'FOLLOW_UP': 'follow_up'
}

# Button configurations for different states
def get_button_configs():
    """Get button configurations using database topics"""
    try:
        from .models import Topic
        topics = Topic.objects.filter(is_active=True).order_by('topic_id')
        topic_buttons = [
            {'id': topic.topic_id, 'label': topic.label, 'type': 'topic'}
            for topic in topics
        ]
    except Exception as e:
        print(f"⚠️ Error fetching topics for buttons from database: {e}")
        # No fallback - return empty list to force database usage
        topic_buttons = []
    
    return {
        'topic_selection': {
            'buttons': topic_buttons,
            'input_enabled': False,
            'message': 'Please select a topic you\'d like to learn about:'
        },
        'follow_up': {
            'buttons': [
                {'id': 'change_topic', 'label': 'Change Topic', 'type': 'action'}
            ],
            'input_enabled': False,
            'message': None  # No additional message needed
        },
        'topic_conversation': {
            'buttons': [
                {'id': 'change_topic', 'label': 'Change Topic', 'type': 'action'}
            ],
            'input_enabled': True,
            'message': None
        }
    }

# Legacy BUTTON_CONFIGS removed - now using database-driven get_button_configs()

def get_topic_keywords(topic_id):
    """Get keywords for a specific topic from database"""
    try:
        from .models import Topic
        topic = Topic.objects.get(topic_id=topic_id, is_active=True)
        return topic.get_keywords_list()  # Returns only active keywords
    except Topic.DoesNotExist:
        print(f"⚠️ Topic '{topic_id}' not found in database")
        return []
    except Exception as e:
        print(f"⚠️ Error fetching keywords from database: {e}")
        return []

def get_all_topic_keywords():
    """Get all keywords from all topics as a flat list from database"""
    all_keywords = []
    try:
        from .models import Topic
        topics = Topic.objects.filter(is_active=True)
        for topic in topics:
            all_keywords.extend(topic.get_keywords_list())
    except Exception as e:
        print(f"⚠️ Error fetching all keywords from database: {e}")
        # No fallback - return empty list to force database usage
    return all_keywords

def find_matching_topics(query_text):
    """Find topics that match keywords in the query text using database"""
    query_lower = query_text.lower()
    matching_topics = []
    
    try:
        from .models import Topic
        topics = Topic.objects.filter(is_active=True)
        
        for topic in topics:
            keywords = topic.get_keywords_list()
            matches = sum(1 for keyword in keywords if keyword.lower() in query_lower)
            if matches > 0:
                matching_topics.append({
                    'topic_id': topic.topic_id,
                    'topic_data': {
                        'label': topic.label,
                        'description': topic.description,
                        'keywords': keywords,
                        'retrieval_strategy': topic.retrieval_strategy
                    },
                    'match_count': matches,
                    'match_ratio': matches / len(keywords) if keywords else 0
                })
    except Exception as e:
        print(f"⚠️ Error fetching topics from database: {e}")
        # No fallback - return empty list to force database usage
        return []
    
    # Sort by match count (descending)
    matching_topics.sort(key=lambda x: x['match_count'], reverse=True)
    return matching_topics

def get_topic_info(topic_id):
    """Get complete information about a topic from database"""
    try:
        from .models import Topic
        topic = Topic.objects.get(topic_id=topic_id, is_active=True)
        return {
            'label': topic.label,
            'description': topic.description,
            'keywords': topic.get_keywords_list(),
            'retrieval_strategy': topic.retrieval_strategy
        }
    except Topic.DoesNotExist:
        print(f"⚠️ Topic '{topic_id}' not found in database")
        return None
    except Exception as e:
        print(f"⚠️ Error fetching topic info from database: {e}")
        return None

def get_topic_retrieval_strategy(topic_id):
    """Get the specialized retrieval strategy for a topic from database"""
    try:
        from .models import Topic
        topic = Topic.objects.get(topic_id=topic_id, is_active=True)
        return topic.retrieval_strategy
    except Topic.DoesNotExist:
        print(f"⚠️ Topic '{topic_id}' not found in database")
        return 'generic'
    except Exception as e:
        print(f"⚠️ Error fetching retrieval strategy from database: {e}")
        return 'generic'

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
