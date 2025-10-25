#!/usr/bin/env python3
"""
Test script for the guided conversation system.
Run this to verify the keyword-based topic filtering works correctly.
"""

import os
import sys
import django

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from chatbot.fast_hybrid_chatbot_together import FastHybridChatbotTogether
from chatbot.topics import CONVERSATION_STATES
import json

def test_guided_conversation():
    """Test the guided conversation system"""
    print("🚀 Testing Guided Conversation System")
    print("=" * 50)
    
    # Initialize chatbot
    chatbot = FastHybridChatbotTogether(use_chroma=True)
    
    # Test 1: Initial state (should show topic selection)
    print("\n📋 Test 1: Initial State")
    result = chatbot.process_guided_conversation("", "message", None)
    print(f"State: {result.get('state')}")
    print(f"Input enabled: {result.get('input_enabled')}")
    print(f"Buttons: {len(result.get('buttons', []))} buttons")
    print(f"Response: {result.get('response', '')[:100]}...")
    
    # Test 2: Topic selection
    print("\n🎯 Test 2: Topic Selection")
    result = chatbot.process_guided_conversation("", "topic_selection", "admissions_enrollment")
    print(f"State: {result.get('state')}")
    print(f"Current topic: {result.get('current_topic')}")
    print(f"Input enabled: {result.get('input_enabled')}")
    print(f"Response: {result.get('response', '')[:100]}...")
    
    # Test 3: Query within topic
    print("\n💬 Test 3: Query within Topic")
    result = chatbot.process_guided_conversation("What are the admission requirements?", "message", None)
    print(f"State: {result.get('state')}")
    print(f"Current topic: {result.get('current_topic')}")
    print(f"Input enabled: {result.get('input_enabled')}")
    print(f"Buttons: {len(result.get('buttons', []))} buttons")
    print(f"Response: {result.get('response', '')[:150]}...")
    
    # Test 4: Follow-up action
    print("\n🔄 Test 4: Follow-up Action")
    result = chatbot.process_guided_conversation("", "action", "ask_another")
    print(f"State: {result.get('state')}")
    print(f"Current topic: {result.get('current_topic')}")
    print(f"Input enabled: {result.get('input_enabled')}")
    print(f"Response: {result.get('response', '')}")
    
    # Test 5: Change topic
    print("\n🔄 Test 5: Change Topic")
    result = chatbot.process_guided_conversation("", "action", "change_topic")
    print(f"State: {result.get('state')}")
    print(f"Current topic: {result.get('current_topic')}")
    print(f"Input enabled: {result.get('input_enabled')}")
    print(f"Buttons: {len(result.get('buttons', []))} buttons")
    
    # Test 6: Auto topic detection
    print("\n🤖 Test 6: Auto Topic Detection")
    chatbot.reset_session()  # Reset to initial state
    result = chatbot.process_guided_conversation("What are the tuition fees for undergraduate programs?", "message", None)
    print(f"State: {result.get('state')}")
    print(f"Current topic: {result.get('current_topic')}")
    print(f"Auto detected topic: {result.get('auto_detected_topic', 'None')}")
    print(f"Response: {result.get('response', '')[:150]}...")
    
    print("\n✅ Guided Conversation Tests Completed!")

def test_topic_filtering():
    """Test the topic-based document filtering"""
    print("\n🔍 Testing Topic-Based Document Filtering")
    print("=" * 50)
    
    # Initialize chatbot
    chatbot = FastHybridChatbotTogether(use_chroma=True)
    
    # Test different topics
    test_queries = [
        ("admissions_enrollment", "What are the admission requirements for new students?"),
        ("fees", "How much is the tuition fee?"),
        ("curriculum", "What subjects are in the computer science program?"),
        ("contact_info", "What is the phone number of the admissions office?")
    ]
    
    for topic_id, query in test_queries:
        print(f"\n📝 Testing topic '{topic_id}' with query: '{query}'")
        
        try:
            # Test topic-filtered retrieval
            docs = chatbot.retrieve_documents_by_topic_keywords(query, topic_id, top_k=3)
            print(f"   Found {len(docs)} documents")
            
            for i, doc in enumerate(docs[:2]):  # Show top 2
                print(f"   Doc {i+1}: {doc.get('filename', 'Unknown')[:50]} (score: {doc.get('relevance', 0):.3f})")
                debug = doc.get('_debug', {})
                if debug:
                    print(f"           Matched keywords: {debug.get('matched_keywords', [])}")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n✅ Topic Filtering Tests Completed!")

def show_topics_info():
    """Display information about all available topics"""
    print("\n📚 Available Topics")
    print("=" * 50)
    
    # TOPICS removed - now using database
    from chatbot.models import Topic
    topics = Topic.objects.filter(is_active=True)
    
    for topic in topics:
        print(f"\n🏷️  {topic.label} ({topic.topic_id})")
        print(f"   Description: {topic.description}")
        keywords = topic.get_keywords_list()
        print(f"   Keywords ({len(keywords)}): {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")

if __name__ == "__main__":
    try:
        show_topics_info()
        test_guided_conversation()
        test_topic_filtering()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
