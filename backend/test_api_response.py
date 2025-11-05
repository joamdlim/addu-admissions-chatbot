#!/usr/bin/env python3
import requests
import json

# Test the guided chat API
base_url = 'http://127.0.0.1:8000/chatbot'
endpoint = f'{base_url}/chat/guided/'

# First select a topic
topic_payload = {
    'user_input': '',
    'action_type': 'topic_selection',
    'action_data': 'programs_courses',
    'session_id': 'test_session'
}

print('🔍 Testing topic selection...')
response = requests.post(endpoint, json=topic_payload)
if response.status_code == 200:
    result = response.json()
    print(f'✅ Topic selected: {result.get("current_topic")}')
    print(f'📝 Response keys: {list(result.keys())}')
else:
    print(f'❌ Topic selection failed: {response.status_code}')
    exit(1)

# Then send a message
message_payload = {
    'user_input': 'What is the Computer Science curriculum?',
    'action_type': 'message',
    'session_id': 'test_session'
}

print('\n🔍 Testing message...')
response = requests.post(endpoint, json=message_payload)
if response.status_code == 200:
    result = response.json()
    print(f'📝 Response keys: {list(result.keys())}')
    if 'sources' in result:
        print(f'📚 Sources found: {len(result["sources"])}')
        if result['sources']:
            print(f'📄 First source keys: {list(result["sources"][0].keys())}')
            print(f'📄 First source sample: {str(result["sources"][0])[:200]}...')
    else:
        print('❌ No sources in response')
else:
    print(f'❌ Message failed: {response.status_code}')
