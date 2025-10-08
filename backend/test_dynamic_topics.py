#!/usr/bin/env python
"""
Test script to verify dynamic topic keyword functionality
"""

import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from chatbot.topics import get_topic_keywords, get_topic_info, find_matching_topics

def test_dynamic_topics():
    print("🧪 Testing dynamic keyword retrieval...")
    
    # Test keyword retrieval
    keywords = get_topic_keywords('programs_courses')
    print(f"✅ Programs keywords count: {len(keywords)}")
    print(f"   First 5 keywords: {keywords[:5]}")
    
    # Test topic info
    topic_info = get_topic_info('programs_courses')
    if topic_info:
        print(f"✅ Topic info label: {topic_info.get('label')}")
        print(f"   Keywords in topic: {len(topic_info.get('keywords', []))}")
    else:
        print("❌ Topic info not found")
    
    # Test topic matching
    matching = find_matching_topics('accountancy program')
    if matching:
        print(f"✅ Topic matching result: {matching[0]['topic_id']}")
        print(f"   Match count: {matching[0]['match_count']}")
    else:
        print("❌ No topic matches found")
    
    # Test specific accountancy keyword
    accountancy_test = find_matching_topics('accountancy')
    if accountancy_test:
        print(f"✅ Accountancy keyword found in topic: {accountancy_test[0]['topic_id']}")
    else:
        print("❌ Accountancy keyword not found")
    
    print("\n🎯 All tests completed!")

if __name__ == "__main__":
    test_dynamic_topics()

