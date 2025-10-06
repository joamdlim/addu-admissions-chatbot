# System Architecture Diagram

## Overall System Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                               │
│                       (React Frontend)                               │
│                                                                       │
│  ┌──────────────┐                           ┌──────────────┐        │
│  │ Free Chat    │                           │ Guided Chat  │        │
│  │ Mode Toggle  │◄──────────────────────────►│ Mode Toggle  │        │
│  └──────────────┘                           └──────────────┘        │
│         │                                            │                │
│         │                                            │                │
└─────────┼────────────────────────────────────────────┼───────────────┘
          │                                            │
          │ HTTP POST                                  │ HTTP POST
          │ /chatbot/chat/                             │ /chatbot/chat/guided/
          ▼                                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DJANGO BACKEND                                  │
│                                                                       │
│  ┌───────────────────────┐            ┌───────────────────────────┐ │
│  │  views.chat_view()    │            │ views.guided_chat_view()  │ │
│  │                       │            │                           │ │
│  │  Calls:               │            │  Calls:                   │ │
│  │  ├─ process_query()   │            │  ├─ process_guided_      │ │
│  │  └─ process_query_    │            │  │   conversation()      │ │
│  │     stream()          │            │  └─ _process_topic_      │ │
│  │                       │            │      query()              │ │
│  └───────────┬───────────┘            └──────────────┬────────────┘ │
│              │                                        │              │
│              │                                        │              │
│              ▼                                        ▼              │
│  ┌───────────────────────┐            ┌───────────────────────────┐ │
│  │ SIMPLE PROMPT         │            │ ENHANCED PROMPT           │ │
│  │ ✅ UNCHANGED          │            │ ✅ MODIFIED               │ │
│  │                       │            │                           │ │
│  │ "You are an           │            │ "You are an ADDU          │ │
│  │  admissions           │            │  Admissions Assistant"    │ │
│  │  assistant..."        │            │                           │ │
│  │                       │            │ + Topic-Specific          │ │
│  │ [Generic Rules]       │            │   Instructions            │ │
│  │                       │            │   ├─ Admissions           │ │
│  │                       │            │   ├─ Programs             │ │
│  │                       │            │   └─ Fees                 │ │
│  │                       │            │                           │ │
│  └───────────┬───────────┘            └──────────────┬────────────┘ │
│              │                                        │              │
└──────────────┼────────────────────────────────────────┼──────────────┘
               │                                        │
               │                                        │
               └────────────────┬───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  TOGETHER AI LLM      │
                    │  (Cloud Service)      │
                    │                       │
                    │  Interprets prompts   │
                    │  and generates        │
                    │  responses            │
                    └───────────────────────┘
```

---

## Guided Chatbot Flow (Enhanced)

```
User selects topic (e.g., "Admissions")
        │
        ▼
process_guided_conversation(action_type='topic_selection')
        │
        ├─ Set session state (topic='admissions_enrollment')
        │
        └─ Return welcome message
                │
                ▼
User asks: "What are the admission requirements?"
        │
        ▼
process_guided_conversation(action_type='message')
        │
        ▼
_process_topic_query(query, topic_id='admissions_enrollment')
        │
        ├─ retrieve_documents_by_topic_specialized()
        │       │
        │       ├─ Filter docs by topic keywords
        │       ├─ TF-IDF + Word2Vec scoring
        │       └─ Return top 3 docs
        │
        ├─ _get_topic_specific_instructions('admissions_enrollment')
        │       │
        │       └─ Returns:
        │           """
        │           === STUDENT TYPE HANDLING ===
        │           - DEFAULT: NEW STUDENTS only
        │           - SPECIFIC: Detect and separate 4 types
        │           - DO NOT MIX: Separate info by type
        │           """
        │
        ├─ Build enhanced prompt:
        │       You are an ADDU Admissions Assistant
        │
        │       [Topic-Specific Instructions] ← INJECTED HERE
        │
        │       GENERAL RESPONSE RULES:
        │       - Be direct and concise
        │       - Start directly with answer
        │       ...
        │
        │       CONTEXT MATCHING:
        │       - Filter by student type
        │       - Prioritize exact matches
        │       ...
        │
        │       <context>
        │       [Retrieved documents]
        │       </context>
        │
        │       <user>
        │       What are the admission requirements?
        │       </user>
        │
        └─ generate_response(prompt) → Together AI
                │
                ▼
        "Admission Requirements for New Students:

        1. **Form 138** (Report Card)
        2. **Birth Certificate**
        ..."
                │
                ▼
        Return response with:
        - state: 'follow_up'
        - buttons: ['Ask Another', 'Change Topic']
        - input_enabled: false
```

---

## Topic-Specific Instructions Flow

```
_process_topic_query(query, topic_id)
        │
        ▼
_get_topic_specific_instructions(topic_id)
        │
        ├─ if topic_id == 'admissions_enrollment':
        │       └─ return """
        │           === STUDENT TYPE HANDLING ===
        │           - DEFAULT: NEW STUDENTS only
        │           - SPECIFIC TYPES: New, Transfer, International, Scholar
        │           - DO NOT MIX: Separate by type
        │           - FOCUS: Requirements, documents, processes
        │           """
        │
        ├─ elif topic_id == 'programs_courses':
        │       └─ return """
        │           === PROGRAM MATCHING ===
        │           - SCOPE: Undergraduate only
        │           - MATCH: Course names & acronyms (BS CS = Computer Science)
        │           - COVERAGE: 60+ programs
        │           - AVOID: Graduate, master's, doctoral
        │
        │           === PROGRAMS WE COVER ===
        │           Arts & Sciences (31), Business (9), Education (6),
        │           Engineering (10), Nursing (1)
        │           """
        │
        ├─ elif topic_id == 'fees':
        │       └─ return """
        │           === FEE INFORMATION ===
        │           - PROGRAM-SPECIFIC: Fees for mentioned program
        │           - INCLUDE: Tuition, misc fees, payment schedules
        │           - DIFFERENTIATE: Different programs = different fees
        │           - SCOPE: Undergraduate only
        │           """
        │
        └─ else:
                └─ return "[Generic instructions]"
```

---

## Document Retrieval Flow (Unchanged)

```
retrieve_documents_by_topic_specialized(query, topic_id)
        │
        ├─ Get topic keywords from topics.py
        │       admissions_enrollment: ['admission', 'requirements',
        │                                'new student', 'transfer', ...]
        │
        ├─ Get all documents from ChromaDB
        │       ▼
        │   ┌─────────────────────────────────────┐
        │   │        CHROMADB                     │
        │   │                                     │
        │   │  Document 1: Admission Guide        │
        │   │  Document 2: Transfer Requirements  │
        │   │  Document 3: BS CS Curriculum       │
        │   │  Document 4: Fee Structure          │
        │   │  ...                                │
        │   └─────────────────────────────────────┘
        │
        ├─ Filter documents by topic keywords
        │       Match keywords in doc metadata
        │       Use word boundaries to avoid substring matches
        │       ▼
        │   Filtered: [Doc 1, Doc 2] (admissions-related)
        │
        ├─ TF-IDF + Word2Vec hybrid scoring
        │       ├─ TF-IDF score (keyword matching)
        │       ├─ Word2Vec score (semantic similarity)
        │       └─ Combined: 0.7 * topic_relevance + 0.3 * semantic
        │
        └─ Return top 3 documents
                ▼
        [
            {id: 'admission_guide', relevance: 0.92, content: '...'},
            {id: 'new_student_req', relevance: 0.87, content: '...'},
            {id: 'documents_list', relevance: 0.81, content: '...'}
        ]
```

---

## Data Flow: User Query → Response

```
USER: "What are the admission requirements?"
  │
  ▼
FRONTEND (React)
  │
  ├─ Detects: Guided Mode
  ├─ Current Topic: admissions_enrollment
  └─ HTTP POST to /chatbot/chat/guided/
      │
      ▼
BACKEND (Django)
  │
  ├─ views.guided_chat_view()
  │   └─ together_chatbot.process_guided_conversation()
  │       └─ _process_topic_query(query, 'admissions_enrollment')
  │
  ├─ retrieve_documents_by_topic_specialized()
  │   ├─ Keywords: ['admission', 'requirements', 'new student', ...]
  │   ├─ ChromaDB query
  │   └─ Returns: 3 relevant docs
  │
  ├─ _get_topic_specific_instructions('admissions_enrollment')
  │   └─ Returns: Admissions-specific instructions
  │
  ├─ Build prompt with:
  │   ├─ Topic instructions (DEFAULT: NEW STUDENTS)
  │   ├─ General rules (direct, concise, etc.)
  │   ├─ Context (3 docs)
  │   └─ User query
  │
  ├─ generate_response() → Together AI
  │   └─ LLM interprets instructions
  │       Filters context to NEW STUDENTS only
  │       Generates focused response
  │
  └─ Returns JSON:
      {
        "response": "Admission Requirements for New Students:...",
        "state": "follow_up",
        "current_topic": "admissions_enrollment",
        "sources": [...],
        "buttons": ["Ask Another", "Change Topic"]
      }
  │
  ▼
FRONTEND (React)
  │
  ├─ Displays bot response
  ├─ Shows "Ask Another" and "Change Topic" buttons
  └─ Disables text input (awaiting button click)
```

---

## Comparison: Free vs Guided Mode

```
┌──────────────────────────────────────────────────────────────────┐
│                       FREE CHAT MODE                             │
│                     (UNCHANGED)                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Query → process_query_stream() → Simple Prompt            │
│                                                                  │
│  Prompt:                                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ "You are an admissions assistant.                          │ │
│  │  Answer using ONLY the provided context.                   │ │
│  │                                                             │ │
│  │  RULES:                                                     │ │
│  │  - Be direct and concise                                   │ │
│  │  - Use simple formatting                                   │ │
│  │  - No introductory phrases                                 │ │
│  │  ..."                                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ✅ Works for general queries                                   │
│  ❌ No topic-specific guidance                                  │
│  ❌ May mix student types                                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                      GUIDED CHAT MODE                            │
│                     (ENHANCED)                                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Topic Selection → User Query → _process_topic_query()          │
│                                                                  │
│  Prompt:                                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ "You are an ADDU Admissions Assistant.                     │ │
│  │                                                             │ │
│  │  [TOPIC-SPECIFIC INSTRUCTIONS] ← INJECTED                  │ │
│  │  ═══════════════════════════════                           │ │
│  │  For Admissions:                                           │ │
│  │    - DEFAULT: NEW STUDENTS only                            │ │
│  │    - SPECIFIC: Detect 4 student types                      │ │
│  │    - DO NOT MIX: Separate by type                          │ │
│  │                                                             │ │
│  │  For Programs:                                             │ │
│  │    - SCOPE: Undergraduate only                             │ │
│  │    - MATCH: Course names & acronyms                        │ │
│  │    - COVERAGE: 60+ programs                                │ │
│  │                                                             │ │
│  │  For Fees:                                                 │ │
│  │    - PROGRAM-SPECIFIC: Fees for mentioned program          │ │
│  │    - DIFFERENTIATE: Different programs = different fees    │ │
│  │                                                             │ │
│  │  GENERAL RESPONSE RULES:                                   │ │
│  │  - Be direct and concise                                   │ │
│  │  - Start directly with answer                              │ │
│  │  ...                                                        │ │
│  │                                                             │ │
│  │  CONTEXT MATCHING:                                         │ │
│  │  - Filter by student type/program                          │ │
│  │  - Prioritize exact matches                                │ │
│  │  ..."                                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ✅ Topic-specific guidance                                     │
│  ✅ Filters by student type                                     │
│  ✅ Matches program variations                                  │
│  ✅ Program-specific fees                                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
addu-admissions-chatbot/
│
├── backend/
│   └── chatbot/
│       ├── fast_hybrid_chatbot_together.py  ← MODIFIED
│       │   ├── _get_topic_specific_instructions()  [NEW]
│       │   ├── _process_topic_query()  [MODIFIED]
│       │   ├── process_query()  [UNCHANGED]
│       │   └── process_query_stream()  [UNCHANGED]
│       │
│       ├── topics.py  [UNCHANGED]
│       │   └── TOPICS dictionary (3 topics defined)
│       │
│       ├── views.py  [UNCHANGED]
│       │   ├── guided_chat_view()
│       │   └── chat_view()
│       │
│       ├── TOPIC_SPECIFIC_INSTRUCTIONS.md  [NEW]
│       └── test_topic_instructions.py  [NEW]
│
├── frontend/
│   └── src/
│       ├── App.jsx  [UNCHANGED]
│       │   └── Mode toggle (Guided vs Free)
│       │
│       └── pages/
│           ├── GuidedChatPage.jsx  [UNCHANGED]
│           └── ChatPage.jsx  [UNCHANGED]
│
├── TOPIC_INSTRUCTIONS_IMPLEMENTATION.md  [NEW]
├── BEFORE_AFTER_COMPARISON.md  [NEW]
├── IMPLEMENTATION_COMPLETE.md  [NEW]
└── ARCHITECTURE_DIAGRAM.md  [NEW] (this file)
```

---

## Key Takeaways

1. **Two Modes, Two Prompts**:

   - Free Chat = Simple generic prompt (unchanged)
   - Guided Chat = Enhanced topic-specific prompt (modified)

2. **Topic-Specific Instructions**:

   - Dynamically injected based on selected topic
   - Provides clear guidance to LLM
   - Maintained in one helper method

3. **No Breaking Changes**:

   - Free chat mode completely unchanged
   - Frontend requires no modifications
   - Backward compatible

4. **Maintainable Design**:
   - Instructions in one place
   - Easy to update per topic
   - Modular and extensible

---

**Status**: ✅ Implementation Complete  
**Date**: October 6, 2025
