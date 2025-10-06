"""
Test script to verify topic-specific instructions are working correctly
in the guided chatbot mode.
"""

import json
import requests

BASE_URL = "http://127.0.0.1:8000/chatbot"

def test_guided_chat(user_input, action_type="message", action_data=None, session_id=None):
    """Send a guided chat request and return the response"""
    endpoint = f"{BASE_URL}/chat/guided/"
    
    payload = {
        "user_input": user_input,
        "action_type": action_type,
        "action_data": action_data,
        "session_id": session_id or f"test_session_{id(user_input)}"
    }
    
    print(f"\n{'='*80}")
    print(f"📤 REQUEST:")
    print(f"   Input: {user_input}")
    print(f"   Action: {action_type}")
    print(f"   Data: {action_data}")
    print(f"{'='*80}")
    
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        
        print(f"\n📥 RESPONSE:")
        print(f"   State: {data.get('state')}")
        print(f"   Topic: {data.get('current_topic')}")
        print(f"   Input Enabled: {data.get('input_enabled')}")
        print(f"\n💬 Bot Response:")
        print(f"   {data.get('response', 'No response')}")
        
        if data.get('sources'):
            print(f"\n📚 Sources ({len(data['sources'])}):")
            for i, source in enumerate(data['sources'][:3], 1):
                print(f"   {i}. {source.get('filename', 'Unknown')} (Relevance: {source.get('relevance', 0):.2f})")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ ERROR: {e}")
        return None

def run_tests():
    """Run test scenarios for topic-specific instructions"""
    
    print("\n" + "="*80)
    print("🧪 TESTING TOPIC-SPECIFIC INSTRUCTIONS")
    print("="*80)
    
    # Test 1: Admissions - Default to New Students
    print("\n\n🔍 TEST 1: Admissions - Default to New Students")
    print("-" * 80)
    session1 = "test_admissions_default"
    
    # Select admissions topic
    test_guided_chat("", "topic_selection", "admissions_enrollment", session1)
    
    # Ask general admissions question (should default to new students)
    test_guided_chat("What are the admission requirements?", "message", None, session1)
    
    
    # Test 2: Admissions - Specific Student Type (Transfer)
    print("\n\n🔍 TEST 2: Admissions - Transfer Students")
    print("-" * 80)
    session2 = "test_admissions_transfer"
    
    # Select admissions topic
    test_guided_chat("", "topic_selection", "admissions_enrollment", session2)
    
    # Ask about transfer students specifically
    test_guided_chat("What are the requirements for transfer students?", "message", None, session2)
    
    
    # Test 3: Programs - Acronym Variation (BS CS)
    print("\n\n🔍 TEST 3: Programs - BS CS (Acronym)")
    print("-" * 80)
    session3 = "test_programs_acronym"
    
    # Select programs topic
    test_guided_chat("", "topic_selection", "programs_courses", session3)
    
    # Ask about BS CS
    test_guided_chat("Tell me about BS CS", "message", None, session3)
    
    
    # Test 4: Programs - Course Name
    print("\n\n🔍 TEST 4: Programs - Computer Science (Name)")
    print("-" * 80)
    session4 = "test_programs_name"
    
    # Select programs topic
    test_guided_chat("", "topic_selection", "programs_courses", session4)
    
    # Ask about Computer Science
    test_guided_chat("What courses are in Computer Science?", "message", None, session4)
    
    
    # Test 5: Fees - Program-Specific
    print("\n\n🔍 TEST 5: Fees - BS Nursing")
    print("-" * 80)
    session5 = "test_fees_nursing"
    
    # Select fees topic
    test_guided_chat("", "topic_selection", "fees", session5)
    
    # Ask about nursing fees
    test_guided_chat("How much is the tuition for BS Nursing?", "message", None, session5)
    
    
    # Test 6: Change Topic
    print("\n\n🔍 TEST 6: Change Topic")
    print("-" * 80)
    
    # Trigger change topic action
    test_guided_chat("", "action", "change_topic", session1)
    
    
    # Test 7: Ask Another Question
    print("\n\n🔍 TEST 7: Ask Another Question (Same Topic)")
    print("-" * 80)
    
    # Continue in admissions topic
    test_guided_chat("", "action", "ask_another", session1)
    test_guided_chat("What documents do I need for admission?", "message", None, session1)
    
    
    print("\n\n" + "="*80)
    print("✅ TESTS COMPLETED")
    print("="*80)
    print("\nREVIEW CHECKLIST:")
    print("1. ✅ Admissions default query → Provided NEW STUDENT info only?")
    print("2. ✅ Transfer students query → Provided TRANSFER info only?")
    print("3. ✅ BS CS acronym → Matched to Computer Science program?")
    print("4. ✅ Computer Science name → Provided curriculum info?")
    print("5. ✅ BS Nursing fees → Provided program-specific fees?")
    print("6. ✅ Change topic → Returned to topic selection?")
    print("7. ✅ Ask another → Stayed in same topic?")
    print("\n")

if __name__ == "__main__":
    print("\n🚀 Starting Topic-Specific Instructions Test")
    print("⚠️  Make sure the Django backend is running on http://127.0.0.1:8000")
    print("⚠️  Run: cd backend && python manage.py runserver\n")
    
    input("Press ENTER to continue...")
    
    run_tests()

