"""
Topic definitions for guided conversation flow with keyword-based document filtering.
All topics and keywords are managed through the database (Topic and TopicKeyword models).
This file contains only the core functions and configuration that interact with the database.
"""

# Conversation states
CONVERSATION_STATES = {
    'TOPIC_SELECTION': 'topic_selection',
    'TOPIC_CONVERSATION': 'topic_conversation',
    'FOLLOW_UP': 'follow_up'
}

def get_button_configs():
    """Get button configurations using database topics only"""
    try:
        from .models import Topic
        topics = Topic.objects.filter(is_active=True).order_by('topic_id')
        topic_buttons = [
            {'id': topic.topic_id, 'label': topic.label, 'type': 'topic'}
            for topic in topics
        ]
    except Exception as e:
        print(f"❌ Error fetching topics for buttons from database: {e}")
        # Return empty list if database fails - no fallback
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

def get_topic_keywords(topic_id):
    """Get keywords for a specific topic from database only"""
    try:
        from .models import Topic
        topic = Topic.objects.get(topic_id=topic_id, is_active=True)
        keywords = topic.get_keywords_list()  # Returns only active keywords
        print(f"📝 Retrieved {len(keywords)} keywords for topic '{topic_id}' from database")
        return keywords
    except Topic.DoesNotExist:
        print(f"❌ Topic '{topic_id}' not found in database")
        return []
    except Exception as e:
        print(f"❌ Error fetching keywords from database for topic '{topic_id}': {e}")
        return []

def get_all_topic_keywords():
    """Get all keywords from all active topics as a flat list from database only"""
    all_keywords = []
    try:
        from .models import Topic
        topics = Topic.objects.filter(is_active=True)
        for topic in topics:
            keywords = topic.get_keywords_list()
            all_keywords.extend(keywords)
            print(f"📝 Added {len(keywords)} keywords from topic '{topic.topic_id}'")
    except Exception as e:
        print(f"❌ Error fetching all keywords from database: {e}")
        return []
    
    print(f"📝 Total keywords retrieved: {len(all_keywords)}")
    return all_keywords

def find_matching_topics(query_text):
    """Find topics that match keywords in the query text using database only"""
    query_lower = query_text.lower()
    matching_topics = []
    
    try:
        from .models import Topic
        topics = Topic.objects.filter(is_active=True)
        
        for topic in topics:
            keywords = topic.get_keywords_list()
            if not keywords:
                continue
                
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
                    'match_ratio': matches / len(keywords)
                })
                print(f"🎯 Topic '{topic.topic_id}' matched {matches}/{len(keywords)} keywords")
                
    except Exception as e:
        print(f"❌ Error finding matching topics from database: {e}")
        return []
    
    # Sort by match count (descending)
    matching_topics.sort(key=lambda x: x['match_count'], reverse=True)
    return matching_topics

def get_topic_info(topic_id):
    """Get complete information about a topic from database only"""
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
        print(f"❌ Topic '{topic_id}' not found in database")
        return None
    except Exception as e:
        print(f"❌ Error fetching topic info from database for '{topic_id}': {e}")
        return None

def get_topic_retrieval_strategy(topic_id):
    """Get the specialized retrieval strategy for a topic from database only"""
    try:
        from .models import Topic
        topic = Topic.objects.get(topic_id=topic_id, is_active=True)
        return topic.retrieval_strategy
    except Topic.DoesNotExist:
        print(f"❌ Topic '{topic_id}' not found in database")
        return 'generic'  # Safe default
    except Exception as e:
        print(f"❌ Error fetching retrieval strategy from database for '{topic_id}': {e}")
        return 'generic'  # Safe default

# Topic-specific retrieval strategy mapping
TOPIC_RETRIEVAL_STRATEGIES = {
    'admissions_specialized': {
        'description': 'Specialized retrieval for admissions, enrollment, requirements, and documents',
        'document_types': ['admission', 'enrollment', 'scholarship', 'policy', 'contact', 'fees', 'other'],  # Added fees and other
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
        'document_types': ['academic', 'curriculum', 'policy'],  # Added policy
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
        'document_types': ['fees', 'financial', 'policy'],  # Added policy
        'metadata_priorities': {
            'filename': 0.3,  # Medium - fee documents may have varied naming
            'keywords': 0.4,  # High priority for fee-related keywords
            'content': 0.3    # Medium for amount and payment term matching
        },
        'amount_extraction': True,       # Extract fee amounts from query
        'payment_term_detection': True,  # Detect payment schedule queries
        'program_level_filtering': True  # Filter by undergraduate/graduate fees
    },
    # NEW: Generic strategy for broader searches
    'generic_enhanced': {
        'description': 'Enhanced generic retrieval that includes all document types',
        'document_types': ['admission', 'enrollment', 'scholarship', 'academic', 'curriculum', 'fees', 'financial', 'policy', 'contact', 'other'],
        'metadata_priorities': {
            'filename': 0.3,
            'keywords': 0.4,
            'content': 0.3
        },
        'fallback_search': True  # Enable fallback to keyword-based search
    }
}

def get_retrieval_strategy_config(strategy_name):
    """Get configuration for a specific retrieval strategy"""
    return TOPIC_RETRIEVAL_STRATEGIES.get(strategy_name, {})

def validate_database_setup():
    """Validate that the database has the required topics and keywords"""
    try:
        from .models import Topic, TopicKeyword
        
        # Check if topics exist
        topics = Topic.objects.filter(is_active=True)
        if not topics.exists():
            print("❌ No active topics found in database!")
            return False
        
        print(f"✅ Found {topics.count()} active topics in database:")
        
        for topic in topics:
            keywords_count = topic.keywords.filter(is_active=True).count()
            print(f"  - {topic.topic_id}: {topic.label} ({keywords_count} keywords)")
            
            if keywords_count == 0:
                print(f"    ⚠️ Warning: Topic '{topic.topic_id}' has no keywords!")
        
        return True
        
    except Exception as e:
        print(f"❌ Database validation failed: {e}")
        return False

# Initialize validation on import
if __name__ != '__main__':
    # Only validate when imported, not when run directly
    try:
        validate_database_setup()
    except Exception as e:
        print(f"⚠️ Could not validate database setup: {e}")