#!/usr/bin/env python3
"""
Test script to verify the BS IT and link hallucination fixes
"""

import os
import sys
import django
import requests
import json
import re

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from chatbot.topics import find_matching_topics, get_topic_keywords

BASE_URL = "http://127.0.0.1:8000/chatbot"

def test_topic_keyword_matching():
    """Test if BS IT queries now match the programs topic"""
    print("\n" + "="*80)
    print("TOPIC KEYWORD MATCHING TEST")
    print("="*80)
    
    test_queries = [
        "curriculum for bs it",
        "curriculum for bsit", 
        "what courses are in bs it",
        "bs it program",
        "bsit curriculum"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        
        # Test topic matching
        matching_topics = find_matching_topics(query)
        
        if matching_topics:
            print(f"   ✅ Found {len(matching_topics)} matching topics:")
            for topic in matching_topics[:2]:
                topic_id = topic['topic_id']
                match_count = topic['match_count']
                print(f"      - {topic_id}: {match_count} keyword matches")
                
                # Check if programs topic is matched
                if topic_id == 'programs_courses':
                    print(f"      ✅ PROGRAMS TOPIC MATCHED! (This should work now)")
        else:
            print(f"   ❌ No matching topics found")

def test_programs_topic_keywords():
    """Test the programs topic keywords"""
    print("\n" + "="*80)
    print("PROGRAMS TOPIC KEYWORDS TEST")
    print("="*80)
    
    try:
        keywords = get_topic_keywords('programs_courses')
        print(f"📝 Programs topic has {len(keywords)} keywords")
        
        # Check for IT-related keywords
        it_related = [kw for kw in keywords if any(term in kw.lower() for term in ['bs it', 'bsit', 'information technology', 'it'])]
        print(f"\n🎯 IT-related keywords found: {len(it_related)}")
        for kw in it_related:
            print(f"   - {kw}")
            
        # Test specific keywords
        test_keywords = ['bs it', 'bsit', 'information technology', 'it']
        for test_kw in test_keywords:
            if test_kw in [kw.lower() for kw in keywords]:
                print(f"   ✅ '{test_kw}' found in keywords")
            else:
                print(f"   ❌ '{test_kw}' NOT found in keywords")
                
    except Exception as e:
        print(f"❌ Error getting topic keywords: {e}")

def test_guided_chat_bs_it():
    """Test guided chat with BS IT queries"""
    print("\n" + "="*80)
    print("GUIDED CHAT BS IT TEST")
    print("="*80)
    
    test_cases = [
        {
            "name": "BS IT Curriculum",
            "query": "curriculum for bs it"
        },
        {
            "name": "BSIT Courses", 
            "query": "what courses are in bsit"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 {test_case['name']}")
        print(f"Query: '{test_case['query']}'")
        
        try:
            response = test_guided_chat_api(test_case['query'])
            if response:
                bot_response = response.get('response', '')
                sources = response.get('sources', [])
                
                print(f"📝 Response length: {len(bot_response)} chars")
                print(f"📄 Sources found: {len(sources)}")
                
                # Check for links in response
                links = extract_links_from_response(bot_response)
                print(f"🔗 Links found: {len(links)}")
                
                # Check for fabricated links
                fabricated_links = []
                for link_text, url in links:
                    if is_likely_fabricated_url(url):
                        fabricated_links.append((link_text, url))
                
                if fabricated_links:
                    print(f"⚠️ WARNING: {len(fabricated_links)} fabricated links detected:")
                    for link_text, url in fabricated_links:
                        print(f"   - [{link_text}]({url})")
                else:
                    print(f"✅ No fabricated links detected")
                
                # Show response preview
                print(f"📋 Response preview: {bot_response[:300]}...")
                
                # Check if response contains IT curriculum info
                if any(term in bot_response.lower() for term in ['information technology', 'programming', 'database', 'networking', 'software']):
                    print(f"✅ Response appears to contain IT-related content")
                else:
                    print(f"❌ Response may not contain IT-related content")
                
            else:
                print("❌ No response received")
                
        except Exception as e:
            print(f"❌ Error testing guided chat: {e}")

def test_guided_chat_api(query, session_id=None):
    """Test the guided chat API"""
    endpoint = f"{BASE_URL}/chat/guided/"
    session_id = session_id or f"test_{hash(query)}"
    
    # First select programs topic
    topic_payload = {
        "session_id": session_id,
        "action_type": "topic_selection",
        "action_data": {"topic_id": "programs_courses"}
    }
    
    try:
        response = requests.post(endpoint, json=topic_payload, timeout=10)
        if response.status_code != 200:
            print(f"❌ Topic selection failed: {response.status_code}")
            return None
        
        # Then send the query
        query_payload = {
            "session_id": session_id,
            "action_type": "message",
            "action_data": {"message": query}
        }
        
        response = requests.post(endpoint, json=query_payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Query failed: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return None

def extract_links_from_response(response_text):
    """Extract markdown links from response text"""
    # Pattern for markdown links: [text](url)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    return re.findall(link_pattern, response_text)

def is_likely_fabricated_url(url):
    """Check if a URL looks fabricated/hallucinated"""
    fabrication_indicators = [
        'example.com',
        'placeholder',
        'your-domain',
        'university.edu',
        'school.edu',
        'addu.edu.ph/fake',
        'addu.edu.ph/example',
        'curriculum-link',
        'program-info',
        'addu.edu.ph/programs',  # Common fabrication pattern
        'addu.edu.ph/curriculum',  # Common fabrication pattern
        'addu.edu.ph/information-technology',  # Specific fabrication
        'addu.edu.ph/bs-it'  # Specific fabrication
    ]
    
    url_lower = url.lower()
    return any(indicator in url_lower for indicator in fabrication_indicators)

def main():
    """Run all tests"""
    print("🚀 Testing BS IT and Link Hallucination Fixes")
    print("="*80)
    
    try:
        # Test topic keyword matching (should work offline)
        test_programs_topic_keywords()
        test_topic_keyword_matching()
        
        # Test guided chat (requires server)
        print("\n" + "="*80)
        print("GUIDED CHAT API TESTS (requires server running)")
        print("="*80)
        print("⚠️  Make sure Django server is running: python manage.py runserver")
        
        user_input = input("\nRun guided chat tests? (y/n): ")
        if user_input.lower() == 'y':
            test_guided_chat_bs_it()
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print("\nExpected results after fixes:")
    print("1. ✅ BS IT queries should match programs topic")
    print("2. ✅ BSIT queries should match programs topic") 
    print("3. ✅ No fabricated links in responses")
    print("4. ✅ BS IT queries should return IT curriculum info")

if __name__ == "__main__":
    main()
