#!/usr/bin/env python3
"""
Test script to analyze the BS IT vs Information Technology issue
and check for link hallucination problems.
"""

import os
import sys
import django
import re
import requests
import json

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from chatbot.models import DocumentMetadata, Topic, TopicKeyword
from chatbot.fast_hybrid_chatbot_together import FastHybridChatbotTogether

BASE_URL = "http://127.0.0.1:8000/chatbot"

def analyze_document_keywords():
    """Analyze document keywords for IT-related terms"""
    print("\n" + "="*80)
    print("DOCUMENT KEYWORD ANALYSIS")
    print("="*80)
    
    # Check for Information Technology documents
    it_docs = DocumentMetadata.objects.filter(keywords__icontains='information technology')
    print(f"\n📄 Documents with 'information technology' keyword: {it_docs.count()}")
    for doc in it_docs[:5]:
        print(f"   - {doc.filename}: {doc.keywords}")
    
    # Check for BS IT documents
    bs_it_docs = DocumentMetadata.objects.filter(keywords__icontains='bs it')
    print(f"\n📄 Documents with 'bs it' keyword: {bs_it_docs.count()}")
    for doc in bs_it_docs[:5]:
        print(f"   - {doc.filename}: {doc.keywords}")
    
    # Check for BSIT documents
    bsit_docs = DocumentMetadata.objects.filter(keywords__icontains='bsit')
    print(f"\n📄 Documents with 'bsit' keyword: {bsit_docs.count()}")
    for doc in bsit_docs[:5]:
        print(f"   - {doc.filename}: {doc.keywords}")
    
    # Check for IT documents
    it_only_docs = DocumentMetadata.objects.filter(keywords__icontains=' it ')
    print(f"\n📄 Documents with ' it ' keyword: {it_only_docs.count()}")
    for doc in it_only_docs[:5]:
        print(f"   - {doc.filename}: {doc.keywords}")

def analyze_topic_keywords():
    """Analyze topic keywords for IT-related terms"""
    print("\n" + "="*80)
    print("TOPIC KEYWORD ANALYSIS")
    print("="*80)
    
    try:
        programs_topic = Topic.objects.get(topic_id='programs_courses')
        keywords = TopicKeyword.objects.filter(topic=programs_topic, is_active=True)
        
        print(f"\n🎯 Programs topic has {keywords.count()} active keywords")
        
        # Check for IT-related keywords
        it_keywords = keywords.filter(keyword__icontains='information technology')
        print(f"\n📝 'information technology' keywords: {it_keywords.count()}")
        for kw in it_keywords:
            print(f"   - {kw.keyword}")
        
        bs_it_keywords = keywords.filter(keyword__icontains='bs it')
        print(f"\n📝 'bs it' keywords: {bs_it_keywords.count()}")
        for kw in bs_it_keywords:
            print(f"   - {kw.keyword}")
        
        bsit_keywords = keywords.filter(keyword__icontains='bsit')
        print(f"\n📝 'bsit' keywords: {bsit_keywords.count()}")
        for kw in bsit_keywords:
            print(f"   - {kw.keyword}")
        
        it_keywords = keywords.filter(keyword__iexact='it')
        print(f"\n📝 'it' (exact) keywords: {it_keywords.count()}")
        for kw in it_keywords:
            print(f"   - {kw.keyword}")
            
    except Topic.DoesNotExist:
        print("❌ Programs topic not found in database")

def test_query_matching():
    """Test how different IT queries are processed"""
    print("\n" + "="*80)
    print("QUERY MATCHING TEST")
    print("="*80)
    
    chatbot = FastHybridChatbotTogether(use_chroma=True, chroma_collection_name="documents")
    
    test_queries = [
        "curriculum for bs it",
        "curriculum for information technology", 
        "curriculum for bsit",
        "curriculum for BS Information Technology",
        "what courses are in bs it",
        "what courses are in information technology"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        
        # Test document retrieval
        try:
            docs = chatbot.retrieve_programs_documents(query, top_k=3)
            print(f"   📄 Retrieved {len(docs)} documents:")
            for i, doc in enumerate(docs[:2]):
                filename = doc.get('filename', 'Unknown')
                relevance = doc.get('relevance', 0)
                print(f"      {i+1}. {filename} (relevance: {relevance:.3f})")
        except Exception as e:
            print(f"   ❌ Error retrieving documents: {e}")

def test_guided_chat_responses():
    """Test guided chat responses for BS IT vs Information Technology"""
    print("\n" + "="*80)
    print("GUIDED CHAT RESPONSE TEST")
    print("="*80)
    
    test_cases = [
        {
            "name": "BS IT Query",
            "query": "curriculum for bs it",
            "expected": "Should return IT curriculum"
        },
        {
            "name": "Information Technology Query", 
            "query": "curriculum for information technology",
            "expected": "Should return IT curriculum"
        },
        {
            "name": "BSIT Query",
            "query": "what courses are in bsit",
            "expected": "Should return IT courses"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 {test_case['name']}")
        print(f"Query: '{test_case['query']}'")
        print(f"Expected: {test_case['expected']}")
        
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
                
                # Show first 200 chars of response
                print(f"📋 Response preview: {bot_response[:200]}...")
                
                if links:
                    print("🔗 Links detected:")
                    for link_text, url in links[:3]:
                        print(f"   - [{link_text}]({url})")
                        # Check if URL looks fabricated
                        if is_likely_fabricated_url(url):
                            print(f"     ⚠️ WARNING: This URL looks fabricated!")
                
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
        'program-info'
    ]
    
    url_lower = url.lower()
    return any(indicator in url_lower for indicator in fabrication_indicators)

def check_document_urls():
    """Check what URLs are actually stored in document metadata"""
    print("\n" + "="*80)
    print("DOCUMENT URL ANALYSIS")
    print("="*80)
    
    # Check if documents have URL fields or URL-like content in keywords
    docs_with_urls = DocumentMetadata.objects.filter(keywords__icontains='http')
    print(f"\n🔗 Documents with HTTP URLs in keywords: {docs_with_urls.count()}")
    
    for doc in docs_with_urls[:5]:
        print(f"   - {doc.filename}")
        # Extract URLs from keywords
        urls = re.findall(r'https?://[^\s,]+', doc.keywords)
        for url in urls:
            print(f"     🔗 {url}")
    
    # Check for common URL patterns
    url_patterns = ['www.', '.com', '.edu', '.ph']
    for pattern in url_patterns:
        docs = DocumentMetadata.objects.filter(keywords__icontains=pattern)
        print(f"\n🌐 Documents with '{pattern}' in keywords: {docs.count()}")

def main():
    """Run all tests"""
    print("🚀 Starting BS IT Issue Analysis")
    print("="*80)
    
    try:
        analyze_document_keywords()
        analyze_topic_keywords()
        test_query_matching()
        check_document_urls()
        
        print("\n" + "="*80)
        print("GUIDED CHAT API TESTS (requires server running)")
        print("="*80)
        print("⚠️  Make sure Django server is running: python manage.py runserver")
        
        user_input = input("\nRun guided chat tests? (y/n): ")
        if user_input.lower() == 'y':
            test_guided_chat_responses()
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey findings will help identify:")
    print("1. Why 'BS IT' queries fail vs 'Information Technology'")
    print("2. Whether topic keywords include proper IT abbreviations")
    print("3. Whether document keywords match query terms")
    print("4. Whether links are being fabricated vs using real document URLs")

if __name__ == "__main__":
    main()
