# ADDU Admissions Chatbot - Guided Conversation System

## Overview

This implementation switches the chatbot from open-ended chat to a **guided conversation flow** with **keyword-based topic filtering** to solve the problem of follow-up questions retrieving wrong documents.

## Problem Solved

**Before**: Follow-up questions like "what about 2nd year" would retrieve wrong documents (e.g., asks about BS IT but gets Early Childhood Education results).

**After**: Users select predefined topics → retrieval filters documents by matching keywords → accurate, contextual responses.

## Architecture

### 1. Topic-Based Structure

```python
TOPICS = {
    'curriculum': {
        'label': 'Program Curriculums',
        'keywords': ['curriculum', 'courses', 'subjects', 'program', 'degree'],
        'description': 'Learn about program curriculums and courses'
    },
    'admissions_enrollment': {
        'label': 'Admissions & Enrollment',
        'keywords': [
            'admission', 'requirements', 'application', 'entrance',
            'enrollment', 'registration', 'enroll', 'register',
            'new student', 'freshman', 'first year',
            'scholar', 'scholarship', 'financial aid', 'grant',
            'transferee', 'transfer', 'shifter',
            'international', 'foreign student', 'foreign'
        ],
        'description': 'Learn about admissions and enrollment for all student types'
    },
    # ... more topics
}
```

### 2. Conversation States

- **TOPIC_SELECTION**: Show topic buttons, disable text input
- **TOPIC_CONVERSATION**: Enable text input within selected topic
- **FOLLOW_UP**: Show "Ask Another Question" or "Change Topic" buttons

### 3. Document Filtering

Documents are filtered by matching their `keywords` metadata field against topic keywords using word boundaries to avoid substring matches.

## API Endpoints

### GET /chatbot/topics/

Returns available topics for guided conversation.

**Response:**

```json
{
    "topics": [
        {
            "id": "admissions_enrollment",
            "label": "Admissions & Enrollment",
            "description": "Learn about admissions and enrollment for all student types",
            "keywords": ["admission", "requirements", "application", ...]
        }
    ],
    "conversation_states": {
        "TOPIC_SELECTION": "topic_selection",
        "TOPIC_CONVERSATION": "topic_conversation",
        "FOLLOW_UP": "follow_up"
    }
}
```

### POST /chatbot/chat/guided/

Handles guided conversation with topic-based filtering.

**Request:**

```json
{
  "user_input": "What are the admission requirements?",
  "action_type": "message", // "message", "topic_selection", "action"
  "action_data": null, // topic_id for topic_selection, action_id for action
  "session_id": "session_123"
}
```

**Response:**

```json
{
    "response": "Here are the admission requirements...",
    "state": "follow_up",
    "buttons": [
        {"id": "ask_another", "label": "Ask Another Question", "type": "action"},
        {"id": "change_topic", "label": "Change Topic", "type": "action"}
    ],
    "input_enabled": false,
    "current_topic": "admissions_enrollment",
    "sources": [...],
    "session_id": "session_123"
}
```

## Conversation Flow

### 1. Initial State

- User visits chatbot
- System shows topic selection buttons
- Text input is disabled

### 2. Topic Selection

- User clicks a topic button
- System sets current topic
- Shows welcome message for the topic
- Enables text input

### 3. Topic Conversation

- User asks questions within the topic
- System filters documents by topic keywords
- Returns relevant answers
- Shows follow-up buttons, disables text input

### 4. Follow-up Actions

- **"Ask Another Question"**: Re-enables text input in same topic
- **"Change Topic"**: Returns to topic selection

### 5. Auto Topic Detection

- If user types instead of selecting topic
- System tries to auto-detect topic from message
- If confident match (≥2 keyword matches), auto-selects topic
- Otherwise, asks user to select topic

## Implementation Details

### Key Components

1. **topics.py**: Topic definitions and utilities
2. **FastHybridChatbotTogether**: Enhanced with session state and topic filtering
3. **views.py**: New guided chat endpoint
4. **urls.py**: New URL routes

### Session State Management

```python
session_state = {
    'current_topic': None,
    'conversation_state': 'topic_selection',
    'session_id': None
}
```

### Document Retrieval

The `retrieve_documents_by_topic_keywords()` method:

1. Gets topic keywords
2. Filters ALL documents by keyword matching
3. Performs semantic search on filtered documents
4. Combines topic relevance (70%) + semantic score (30%)
5. Returns top-k results

### Keyword Matching

Uses regex word boundaries to avoid substring matches:

```python
pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
```

This prevents "arch" from matching "Anthropology".

## Configuration

### Adding New Topics

1. Add to `TOPICS` dictionary in `topics.py`
2. Define appropriate keywords
3. Keywords should match document metadata `keywords` field

### Customizing Button Behavior

Modify `BUTTON_CONFIGS` in `topics.py`:

```python
BUTTON_CONFIGS = {
    'topic_selection': {
        'buttons': [...],
        'input_enabled': False,
        'message': 'Please select a topic...'
    }
}
```

## Testing

### Run Tests

```bash
cd backend
python chatbot/test_guided_conversation.py
```

### API Testing

```bash
python chatbot/api_usage_example.py
```

### Manual Testing

1. **GET Topics**: `curl http://localhost:8000/chatbot/topics/`

2. **Guided Chat**:

```bash
curl -X POST http://localhost:8000/chatbot/chat/guided/ \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "",
    "action_type": "topic_selection",
    "action_data": "admissions_enrollment",
    "session_id": "test123"
  }'
```

## Benefits

1. **Accurate Retrieval**: Topic filtering ensures relevant documents
2. **Better UX**: Guided flow prevents user confusion
3. **Contextual Responses**: Maintains topic context throughout conversation
4. **Flexible**: Supports both guided and auto-detection modes
5. **Scalable**: Easy to add new topics and keywords

## Technical Requirements Met

✅ **Keep both NLP models**: TF-IDF + Word2Vec still used for semantic scoring  
✅ **Use global chatbot instance**: `together_chatbot` instance reused  
✅ **Keywords configurable**: Easy to modify in `topics.py`  
✅ **Document keywords matching**: Matches against `DocumentMetadata.keywords` field  
✅ **Session state management**: Tracks topic and conversation state  
✅ **API response format**: Returns state, buttons, input_enabled, current_topic

## Future Enhancements

1. **Multi-language support**: Add topic translations
2. **Dynamic topics**: Auto-generate topics from document analysis
3. **User preferences**: Remember preferred topics per user
4. **Analytics**: Track topic popularity and success rates
5. **Voice interface**: Add voice commands for topic selection

## Troubleshooting

### Common Issues

1. **No documents found for topic**: Check if document keywords match topic keywords
2. **Auto-detection not working**: Increase keyword coverage in topic definitions
3. **Session state lost**: Ensure session_id is passed consistently

### Debug Mode

Enable debug output by checking console logs for:

- `🎯 Topic-filtered retrieval`
- `📝 Topic keywords`
- `✅ Top X topic-filtered results`

### Keyword Matching Debug

The system shows matched keywords in debug output:

```
Matched keywords: ['admission', 'requirements', 'application']
```

Use this to verify topic-document matching is working correctly.
