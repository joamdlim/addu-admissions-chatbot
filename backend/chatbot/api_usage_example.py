#!/usr/bin/env python3
"""
API Usage Examples for the Guided Conversation System

This shows how to use the new guided conversation endpoints:
- GET /chatbot/topics/ - Get available topics
- POST /chatbot/chat/guided/ - Guided conversation with topic filtering
"""

import requests
import json

# Base URL (adjust as needed)
BASE_URL = "http://localhost:8000/chatbot"

def test_get_topics():
    """Test the topics endpoint"""
    print("🔍 Testing GET /chatbot/topics/")
    
    try:
        response = requests.get(f"{BASE_URL}/topics/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {len(data['topics'])} topics:")
            for topic in data['topics']:
                print(f"   - {topic['label']} ({topic['id']})")
                print(f"     {topic['description']}")
                print(f"     Keywords: {len(topic['keywords'])} keywords")
            return data
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def test_guided_conversation():
    """Test the guided conversation flow"""
    print("\n💬 Testing Guided Conversation Flow")
    print("=" * 50)
    
    session_id = "test_session_123"
    
    # Step 1: Initial request (should show topic selection)
    print("\n1️⃣ Initial request (topic selection)")
    response = requests.post(f"{BASE_URL}/chat/guided/", json={
        "user_input": "",
        "action_type": "message",
        "session_id": session_id
    })
    
    if response.status_code == 200:
        data = response.json()
        print(f"   State: {data.get('state')}")
        print(f"   Input enabled: {data.get('input_enabled')}")
        print(f"   Buttons: {len(data.get('buttons', []))}")
        print(f"   Response: {data.get('response', '')[:100]}...")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return
    
    # Step 2: Select a topic
    print("\n2️⃣ Select 'Admissions & Enrollment' topic")
    response = requests.post(f"{BASE_URL}/chat/guided/", json={
        "user_input": "",
        "action_type": "topic_selection",
        "action_data": "admissions_enrollment",
        "session_id": session_id
    })
    
    if response.status_code == 200:
        data = response.json()
        print(f"   State: {data.get('state')}")
        print(f"   Current topic: {data.get('current_topic')}")
        print(f"   Input enabled: {data.get('input_enabled')}")
        print(f"   Response: {data.get('response', '')[:100]}...")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return
    
    # Step 3: Ask a question within the topic
    print("\n3️⃣ Ask question within topic")
    response = requests.post(f"{BASE_URL}/chat/guided/", json={
        "user_input": "What are the admission requirements for new students?",
        "action_type": "message",
        "session_id": session_id
    })
    
    if response.status_code == 200:
        data = response.json()
        print(f"   State: {data.get('state')}")
        print(f"   Current topic: {data.get('current_topic')}")
        print(f"   Input enabled: {data.get('input_enabled')}")
        print(f"   Buttons: {len(data.get('buttons', []))}")
        print(f"   Response: {data.get('response', '')[:200]}...")
        
        # Show sources if available
        sources = data.get('sources', [])
        if sources:
            print(f"   Sources: {len(sources)} documents")
            for i, source in enumerate(sources[:2]):
                print(f"     {i+1}. {source.get('filename', 'Unknown')[:50]}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return
    
    # Step 4: Follow-up action
    print("\n4️⃣ Ask another question")
    response = requests.post(f"{BASE_URL}/chat/guided/", json={
        "user_input": "",
        "action_type": "action",
        "action_data": "ask_another",
        "session_id": session_id
    })
    
    if response.status_code == 200:
        data = response.json()
        print(f"   State: {data.get('state')}")
        print(f"   Input enabled: {data.get('input_enabled')}")
        print(f"   Response: {data.get('response', '')}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
    
    # Step 5: Change topic
    print("\n5️⃣ Change topic")
    response = requests.post(f"{BASE_URL}/chat/guided/", json={
        "user_input": "",
        "action_type": "action",
        "action_data": "change_topic",
        "session_id": session_id
    })
    
    if response.status_code == 200:
        data = response.json()
        print(f"   State: {data.get('state')}")
        print(f"   Current topic: {data.get('current_topic')}")
        print(f"   Input enabled: {data.get('input_enabled')}")
        print(f"   Buttons: {len(data.get('buttons', []))}")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

def test_auto_topic_detection():
    """Test automatic topic detection from user message"""
    print("\n🤖 Testing Auto Topic Detection")
    print("=" * 40)
    
    test_messages = [
        "What are the tuition fees?",
        "How do I apply for admission?",
        "What subjects are in the computer science curriculum?",
        "What's the phone number of the registrar?"
    ]
    
    for message in test_messages:
        print(f"\n📝 Message: '{message}'")
        response = requests.post(f"{BASE_URL}/chat/guided/", json={
            "user_input": message,
            "action_type": "message",
            "session_id": f"auto_test_{hash(message)}"
        })
        
        if response.status_code == 200:
            data = response.json()
            auto_topic = data.get('auto_detected_topic')
            current_topic = data.get('current_topic')
            
            if auto_topic:
                print(f"   ✅ Auto-detected: {auto_topic} ({current_topic})")
            else:
                print(f"   ❓ No auto-detection, state: {data.get('state')}")
        else:
            print(f"   ❌ Error: {response.status_code}")

if __name__ == "__main__":
    print("🚀 Testing ADDU Admissions Chatbot - Guided Conversation API")
    print("=" * 60)
    
    # Test endpoints
    topics_data = test_get_topics()
    
    if topics_data:
        test_guided_conversation()
        test_auto_topic_detection()
        print("\n✅ All API tests completed!")
    else:
        print("\n❌ Cannot proceed without topics data")
