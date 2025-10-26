#!/usr/bin/env python3

import os
import sys
import django

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')
django.setup()

from chatbot.fast_hybrid_chatbot_together import FastHybridChatbotTogether

def test_specific_bscs_fees_query():
    """Test the specific failing BSCS fees query"""
    print("=== Testing Specific BSCS Fees Query ===")
    
    try:
        # Initialize chatbot
        chatbot = FastHybridChatbotTogether(
            use_chroma=True, 
            chroma_collection_name='documents',
            use_hybrid_topic_retrieval=True
        )
        
        # Test the specific failing query
        failing_query = "what is the tuition fee for the first year 2nd sem for bscs"
        print(f"\nTesting query: '{failing_query}'")
        
        # Process the query
        response = chatbot._process_topic_query(
            query=failing_query,
            topic_id="fees",
            conversation_history=[],
            user_id="test_user"
        )
        
        print(f"\nResponse: {response['response'][:500]}...")
        print(f"\nSources ({len(response['sources'])}):")
        for i, source in enumerate(response['sources'], 1):
            print(f"{i}. {source['filename']} (Relevance: {source['relevance']:.1f}%)")
            if 'SAS-FEES-3.csv' in source['filename']:
                print("   ✓ SAS-FEES-3.csv found!")
            
        # Test the working query for comparison
        working_query = "give me bscs fees"
        print(f"\n\nTesting working query: '{working_query}'")
        
        response2 = chatbot._process_topic_query(
            query=working_query,
            topic_id="fees",
            conversation_history=[],
            user_id="test_user"
        )
        
        print(f"\nResponse: {response2['response'][:500]}...")
        print(f"\nSources ({len(response2['sources'])}):")
        for i, source in enumerate(response2['sources'], 1):
            print(f"{i}. {source['filename']} (Relevance: {source['relevance']:.1f}%)")
            if 'SAS-FEES-3.csv' in source['filename']:
                print("   ✓ SAS-FEES-3.csv found!")
                
        # Test intermediate query
        intermediate_query = "bscs tuition fees first year second semester"
        print(f"\n\nTesting intermediate query: '{intermediate_query}'")
        
        response3 = chatbot._process_topic_query(
            query=intermediate_query,
            topic_id="fees",
            conversation_history=[],
            user_id="test_user"
        )
        
        print(f"\nResponse: {response3['response'][:500]}...")
        print(f"\nSources ({len(response3['sources'])}):")
        for i, source in enumerate(response3['sources'], 1):
            print(f"{i}. {source['filename']} (Relevance: {source['relevance']:.1f}%)")
            if 'SAS-FEES-3.csv' in source['filename']:
                print("   ✓ SAS-FEES-3.csv found!")
                
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_specific_bscs_fees_query()
