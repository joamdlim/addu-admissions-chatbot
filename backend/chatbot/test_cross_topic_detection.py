"""
Test script to verify cross-topic detection and handling
"""

import json
import requests

BASE_URL = "http://127.0.0.1:8000/chatbot"

def test_cross_topic_scenarios():
    """Test various cross-topic scenarios"""
    
    print("\n" + "="*80)
    print("🧪 TESTING CROSS-TOPIC DETECTION")
    print("="*80)
    
    # Test scenarios: (current_topic, user_query, expected_behavior)
    test_cases = [
        # User in Programs, asks about Admissions
        {
            "name": "Programs → Admissions",
            "current_topic": "programs_courses", 
            "query": "What are the admission requirements?",
            "expected": "Should detect admissions topic and suggest changing topic"
        },
        
        # User in Admissions, asks about Programs
        {
            "name": "Admissions → Programs", 
            "current_topic": "admissions_enrollment",
            "query": "Tell me about BS Computer Science",
            "expected": "Should detect programs topic and suggest changing topic"
        },
        
        # User in Programs, asks about Fees
        {
            "name": "Programs → Fees",
            "current_topic": "programs_courses", 
            "query": "How much is the tuition for BS Nursing?",
            "expected": "Should detect fees topic and suggest changing topic"
        },
        
        # User in Fees, asks about Admissions
        {
            "name": "Fees → Admissions",
            "current_topic": "fees",
            "query": "What documents do I need for admission?",
            "expected": "Should detect admissions topic and suggest changing topic"
        },
        
        # User in Admissions, asks about Fees
        {
            "name": "Admissions → Fees",
            "current_topic": "admissions_enrollment",
            "query": "What are the tuition fees?",
            "expected": "Should detect fees topic and suggest changing topic"
        },
        
        # User in Fees, asks about Programs
        {
            "name": "Fees → Programs",
            "current_topic": "fees",
            "query": "What courses are in Computer Science?",
            "expected": "Should detect programs topic and suggest changing topic"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n🔍 TEST {i}: {test_case['name']}")
        print("-" * 80)
        print(f"Current Topic: {test_case['current_topic']}")
        print(f"User Query: \"{test_case['query']}\"")
        print(f"Expected: {test_case['expected']}")
        
        # Test the cross-topic detection
        result = test_guided_chat_cross_topic(
            test_case['query'], 
            test_case['current_topic']
        )
        
        if result:
            print(f"\n✅ Response received")
            print(f"📝 Bot Response: {result.get('response', 'No response')[:200]}...")
            
            # Check if it detected cross-topic
            if "Change Topic" in result.get('response', ''):
                print("✅ Cross-topic detection working!")
            else:
                print("⚠️ Cross-topic detection may not be working")
        else:
            print("❌ No response received")

def test_guided_chat_cross_topic(user_input, current_topic, session_id=None):
    """Test cross-topic detection in guided chat"""
    endpoint = f"{BASE_URL}/chat/guided/"
    
    # First, set the current topic
    session_id = session_id or f"cross_topic_test_{id(user_input)}"
    
    # Set topic
    set_topic_response = requests.post(endpoint, json={
        "user_input": "",
        "action_type": "topic_selection", 
        "action_data": current_topic,
        "session_id": session_id
    })
    
    if not set_topic_response.ok:
        print(f"❌ Failed to set topic: {set_topic_response.status_code}")
        return None
    
    # Now ask the cross-topic question
    response = requests.post(endpoint, json={
        "user_input": user_input,
        "action_type": "message",
        "action_data": None,
        "session_id": session_id
    })
    
    if response.ok:
        return response.json()
    else:
        print(f"❌ Request failed: {response.status_code}")
        return None

def test_same_topic_queries():
    """Test that same-topic queries still work normally"""
    
    print("\n\n" + "="*80)
    print("🧪 TESTING SAME-TOPIC QUERIES (Should work normally)")
    print("="*80)
    
    same_topic_tests = [
        {
            "topic": "admissions_enrollment",
            "query": "What are the requirements for new students?",
            "expected": "Should work normally (admissions query in admissions topic)"
        },
        {
            "topic": "programs_courses", 
            "query": "What courses are in the Computer Science program?",
            "expected": "Should work normally (programs query in programs topic)"
        },
        {
            "topic": "fees",
            "query": "What are the tuition fees for BS Nursing?",
            "expected": "Should work normally (fees query in fees topic)"
        }
    ]
    
    for i, test in enumerate(same_topic_tests, 1):
        print(f"\n\n🔍 SAME-TOPIC TEST {i}: {test['topic']}")
        print("-" * 80)
        print(f"Query: \"{test['query']}\"")
        print(f"Expected: {test['expected']}")
        
        result = test_guided_chat_cross_topic(test['query'], test['topic'])
        
        if result:
            print(f"✅ Response received")
            print(f"📝 Bot Response: {result.get('response', 'No response')[:200]}...")
            
            # Check if it's a normal response (not cross-topic detection)
            if "Change Topic" not in result.get('response', ''):
                print("✅ Same-topic query working normally!")
            else:
                print("⚠️ Same-topic query incorrectly detected as cross-topic")
        else:
            print("❌ No response received")

def run_all_tests():
    """Run all cross-topic detection tests"""
    
    print("\n🚀 Starting Cross-Topic Detection Tests")
    print("⚠️  Make sure the Django backend is running on http://127.0.0.1:8000")
    print("⚠️  Run: cd backend && python manage.py runserver\n")
    
    input("Press ENTER to continue...")
    
    # Test cross-topic scenarios
    test_cross_topic_scenarios()
    
    # Test same-topic scenarios
    test_same_topic_queries()
    
    print("\n\n" + "="*80)
    print("✅ CROSS-TOPIC DETECTION TESTS COMPLETED")
    print("="*80)
    print("\nREVIEW CHECKLIST:")
    print("1. ✅ Cross-topic queries detected and redirected")
    print("2. ✅ Same-topic queries work normally")
    print("3. ✅ Helpful guidance provided for topic switching")
    print("4. ✅ No false positives in same-topic queries")
    print("\n")

if __name__ == "__main__":
    run_all_tests()
