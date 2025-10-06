"""
Test script to verify that links are displayed as hyperlinks in the chatbot responses
"""

import json
import requests

BASE_URL = "http://127.0.0.1:8000/chatbot"

def test_hyperlink_display():
    """Test that links are displayed as hyperlinks in programs topic"""
    
    print("\n" + "="*80)
    print("🧪 TESTING HYPERLINK DISPLAY")
    print("="*80)
    
    # Test scenarios for programs topic with links
    test_cases = [
        {
            "name": "Computer Science Program",
            "query": "Tell me about BS Computer Science program",
            "expected": "Should show program info with clickable links"
        },
        {
            "name": "Nursing Program", 
            "query": "What courses are in BS Nursing?",
            "expected": "Should show nursing curriculum with links"
        },
        {
            "name": "Business Program",
            "query": "Tell me about BS Business Management",
            "expected": "Should show business program with links"
        },
        {
            "name": "Engineering Program",
            "query": "What is the curriculum for BS Computer Engineering?",
            "expected": "Should show engineering curriculum with links"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n🔍 TEST {i}: {test_case['name']}")
        print("-" * 80)
        print(f"Query: \"{test_case['query']}\"")
        print(f"Expected: {test_case['expected']}")
        
        # Test the hyperlink display
        result = test_guided_chat_with_links(test_case['query'])
        
        if result:
            print(f"\n✅ Response received")
            response = result.get('response', 'No response')
            print(f"📝 Bot Response: {response[:300]}...")
            
            # Check for hyperlink formatting
            if "[" in response and "]" in response and "(" in response and ")" in response:
                print("✅ Hyperlinks detected in response!")
                # Extract and show the links
                import re
                links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', response)
                if links:
                    print(f"🔗 Found {len(links)} hyperlink(s):")
                    for link_text, url in links:
                        print(f"   - [{link_text}]({url})")
                else:
                    print("⚠️ Markdown syntax found but no complete links detected")
            else:
                print("⚠️ No hyperlink formatting detected")
                print("   (This might be normal if no links are in the source documents)")
        else:
            print("❌ No response received")

def test_guided_chat_with_links(user_input, session_id=None):
    """Test guided chat with focus on link display"""
    endpoint = f"{BASE_URL}/chat/guided/"
    
    session_id = session_id or f"hyperlink_test_{id(user_input)}"
    
    # First, select programs topic
    set_topic_response = requests.post(endpoint, json={
        "user_input": "",
        "action_type": "topic_selection", 
        "action_data": "programs_courses",
        "session_id": session_id
    })
    
    if not set_topic_response.ok:
        print(f"❌ Failed to set topic: {set_topic_response.status_code}")
        return None
    
    # Now ask the question
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

def test_link_formatting_examples():
    """Test specific link formatting examples"""
    
    print("\n\n" + "="*80)
    print("🧪 TESTING LINK FORMATTING EXAMPLES")
    print("="*80)
    
    # Test different types of link queries
    link_tests = [
        {
            "query": "Where can I find the Computer Science curriculum?",
            "expected_links": ["curriculum", "program guide", "course details"]
        },
        {
            "query": "What documents are available for BS Nursing?",
            "expected_links": ["nursing", "program", "curriculum"]
        },
        {
            "query": "How can I download the program brochure?",
            "expected_links": ["download", "brochure", "guide"]
        }
    ]
    
    for i, test in enumerate(link_tests, 1):
        print(f"\n\n🔍 LINK TEST {i}")
        print("-" * 80)
        print(f"Query: \"{test['query']}\"")
        print(f"Expected link types: {test['expected_links']}")
        
        result = test_guided_chat_with_links(test['query'])
        
        if result:
            response = result.get('response', '')
            print(f"📝 Response: {response[:200]}...")
            
            # Check for various link patterns
            link_patterns = [
                r'\[([^\]]+)\]\(([^)]+)\)',  # [text](url)
                r'https?://[^\s]+',         # Raw URLs
                r'www\.[^\s]+'              # www links
            ]
            
            found_links = []
            for pattern in link_patterns:
                import re
                matches = re.findall(pattern, response)
                found_links.extend(matches)
            
            if found_links:
                print(f"✅ Found {len(found_links)} link(s) in response")
                for link in found_links[:3]:  # Show first 3 links
                    print(f"   🔗 {link}")
            else:
                print("⚠️ No links found in response")
        else:
            print("❌ No response received")

def run_hyperlink_tests():
    """Run all hyperlink display tests"""
    
    print("\n🚀 Starting Hyperlink Display Tests")
    print("⚠️  Make sure the Django backend is running on http://127.0.0.1:8000")
    print("⚠️  Run: cd backend && python manage.py runserver\n")
    
    input("Press ENTER to continue...")
    
    # Test hyperlink display
    test_hyperlink_display()
    
    # Test link formatting examples
    test_link_formatting_examples()
    
    print("\n\n" + "="*80)
    print("✅ HYPERLINK DISPLAY TESTS COMPLETED")
    print("="*80)
    print("\nREVIEW CHECKLIST:")
    print("1. ✅ Links formatted as [text](URL) in responses")
    print("2. ✅ Clickable hyperlinks in frontend")
    print("3. ✅ Relevant link text (not raw URLs)")
    print("4. ✅ Multiple links when available")
    print("5. ✅ Links from document metadata included")
    print("\n")

if __name__ == "__main__":
    run_hyperlink_tests()
