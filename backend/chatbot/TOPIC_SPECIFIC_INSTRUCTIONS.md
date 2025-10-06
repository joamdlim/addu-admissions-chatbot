# Topic-Specific Instructions for ADDU Admissions Chatbot

## Overview

This document describes the specialized instructions implemented in the **Guided Chatbot Mode** to ensure accurate, context-appropriate responses for different topics in the ADDU Admissions Assistant.

**IMPORTANT**: These instructions are **ONLY applied to the Guided Chatbot Mode** (topic-based conversation). The Free Chat Mode remains unchanged with generic prompts.

---

## Implementation Location

**File**: `backend/chatbot/fast_hybrid_chatbot_together.py`

**Methods Modified**:
1. `_process_topic_query()` - Updated system prompt to include topic-specific instructions
2. `_get_topic_specific_instructions()` - NEW method that returns specialized instructions per topic

**Methods Unchanged** (Free Chat Mode):
- `process_query()` - Open-ended chat (simple prompt)
- `process_query_stream()` - Open-ended streaming chat (simple prompt)

---

## Topic-Specific Instructions

### 1. General Admissions and Enrollment (`admissions_enrollment`)

#### Student Type Handling

**DEFAULT BEHAVIOR**:
- If user asks generally about "admissions" or "requirements" **WITHOUT** specifying student type → Provide information for **NEW STUDENTS ONLY**

**SPECIFIC STUDENT TYPES**:
When user mentions a specific type, provide information ONLY for that type:

| Student Type | Keywords | Description |
|--------------|----------|-------------|
| **NEW STUDENTS** | first-time, freshmen, incoming students | First-time college students |
| **TRANSFER STUDENTS** | transferring, shifters, lateral entry | Students from other institutions |
| **INTERNATIONAL STUDENTS** | foreign students, overseas, non-Filipino | Students from abroad |
| **SCHOLAR STUDENTS** | scholarship, financial aid, grant holders | Scholarship recipients |

**CRITICAL RULES**:
- ❌ **DO NOT MIX** information from different student types in a single response
- ✅ **FOCUS** on admission requirements, documents, processes specific to identified type
- ✅ If context contains multiple types but user asked about ONE, filter accordingly

---

### 2. Programs and Courses (`programs_courses`)

#### Program Matching

**BASE RESPONSES** on specific COURSE NAME and ACRONYM mentioned:

**Match Variations**:
- Course name: "Computer Science" → BS Computer Science
- Acronyms: "BSCS", "BS CS", "BS COMSCI" → Same program
- Handle spacing variations in acronyms

**SCOPE**:
- ✅ **INCLUDE**: Undergraduate programs ONLY (Bachelor's degrees, BS, BA)
- ✅ **INCLUDE**: Whole document context for matched program
- ❌ **AVOID**: Graduate programs, master's, doctoral, senior high school

#### Programs Covered

**ARTS AND SCIENCES** (31 programs):
- AB Anthropology (5 tracks: Academic Research, Community Development, IP Education, Medical Anthropology, Pre-Law)
- AB Communication
- AB Development Studies
- AB Economics
- AB English Language
- AB Interdisciplinary Studies (5 minors: Language & Literature, Media & Business, Media & Philosophy, Media & Technology, Philosophy & Theology)
- AB International Studies (2 majors: American Studies, Asian Studies)
- AB Islamic Studies
- AB Philosophy
- AB Political Studies
- AB Psychology
- AB Sociology
- BS Biology (2 tracks: General Biology, Medical Biology)
- BS Chemistry
- BS Computer Science
- BS Data Science
- BS Environmental Science
- BS Information Systems
- BS Information Technology
- BS Mathematics
- BS Social Work

**BUSINESS AND GOVERNANCE** (9 programs):
- Bachelor in Public Management (BPM)
- BS Accountancy (BSA)
- BS Management Accounting (BSMA)
- BS Business Management (BSBM)
- BS Entrepreneurship (2 tracks: General, Agri-Business)
- BS Finance (BSFIN)
- BS Human Resource Development and Management (BSHRDM)
- BS Marketing

**EDUCATION** (6 programs):
- Bachelor of Early Childhood Education (BECE)
- Bachelor of Elementary Education (BEED)
- BS Secondary Education - English (BSED-English)
- BS Secondary Education - Mathematics (BSED-Math)
- BS Secondary Education - Science (BSED-Science)
- BS Secondary Education - Social Studies (BSED-SS)

**ENGINEERING AND ARCHITECTURE** (10 programs):
- BS Aerospace Engineering
- BS Architecture
- BS Chemical Engineering
- BS Civil Engineering
- BS Computer Engineering
- BS Electrical Engineering
- BS Electronics Engineering
- BS Industrial Engineering
- BS Mechanical Engineering
- BS Robotics Engineering

**NURSING** (1 program):
- BS Nursing (BSN)

**TOTAL**: 60+ Undergraduate Programs

---

### 3. Fees (`fees`)

#### Fee Information Handling

**BASE RESPONSES** on specific COURSE/PROGRAM mentioned:

**MATCH**:
- Provide fee information specific to the program user asked about
- Different programs may have different fee structures

**INCLUDE**:
- Tuition fees
- Miscellaneous fees
- Payment schedules
- Installment options
- Program-specific fees

**DIFFERENTIATE**:
- Engineering programs may have different fees than Arts programs
- Nursing may have clinical/lab fees
- Architecture may have studio fees

**SCOPE**:
- ✅ Undergraduate program fees only
- ❌ Avoid graduate program fees

---

## General Response Rules (All Topics)

These rules apply to ALL topics in guided mode:

### Formatting
- ✅ Be direct and concise
- ✅ Start directly with the answer
- ✅ Use bullet points for lists
- ✅ Use numbered lists for step-by-step processes
- ✅ Bold important terms and amounts
- ❌ No introductory phrases like "Based on the documents"
- ❌ No closing phrases like "I hope this helps"

### Context Matching
- ✅ Only use information that directly matches the user's specific query
- ✅ If context contains multiple student types/programs but user asked about ONE, filter accordingly
- ✅ Prioritize exact matches over general information
- ✅ If information not available, state clearly what specific information is missing

---

## Architecture Diagram

```
User Query → Guided Chat Endpoint → process_guided_conversation()
                                              ↓
                              _process_topic_query(query, topic_id)
                                              ↓
                        _get_topic_specific_instructions(topic_id)
                                              ↓
                              [Topic-Specific Instructions]
                                       ↓         ↓         ↓
                            Admissions   Programs    Fees
                                       ↓
                              System Prompt Built
                                       ↓
                              Together AI LLM
                                       ↓
                              Formatted Response
```

---

## Testing Examples

### Example 1: Admissions (Default to New Students)

**User Query**: "What are the admission requirements?"

**Expected Behavior**:
- Detect no specific student type mentioned
- Default to NEW STUDENTS
- Provide NEW STUDENT requirements only
- DO NOT include transfer/international/scholar requirements

---

### Example 2: Admissions (Specific Student Type)

**User Query**: "What are the requirements for transfer students?"

**Expected Behavior**:
- Detect "transfer students" keyword
- Provide TRANSFER STUDENT requirements only
- DO NOT include new/international/scholar requirements

---

### Example 3: Programs (Acronym Variation)

**User Query**: "Tell me about BS CS"

**Expected Behavior**:
- Match "BS CS" to "BS Computer Science"
- Provide full curriculum information
- Include course details, requirements, etc.

---

### Example 4: Programs (Course Name)

**User Query**: "What courses are in Computer Science?"

**Expected Behavior**:
- Match "Computer Science" to BS Computer Science program
- Provide course list and curriculum
- Focus on undergraduate program only

---

### Example 5: Fees (Program-Specific)

**User Query**: "How much is the tuition for BS Nursing?"

**Expected Behavior**:
- Match "BS Nursing" program
- Provide BSN-specific fees
- Include any nursing-specific fees (clinical, lab)
- DO NOT provide generic fees for other programs

---

## Code Implementation

### System Prompt Structure

```python
prompt = f"""<|system|>
You are an ADDU (Ateneo de Davao University) Admissions Assistant. 
You provide accurate, helpful information based strictly on the provided context documents.

{topic_specific_instructions}  # <-- Dynamically inserted based on topic

GENERAL RESPONSE RULES:
- Be direct and concise
- Start directly with the answer
...

CONTEXT MATCHING:
- Only use information that directly matches the user's specific query
...
</|system|>

<|context|>
{doc_context}
</|context|>

<|user|>
{query}
</|user|>

<|assistant|>
"""
```

### Topic-Specific Instructions Method

```python
def _get_topic_specific_instructions(self, topic_id: str) -> str:
    """Get specialized instructions based on the topic"""
    if topic_id == 'admissions_enrollment':
        return """[Detailed Admissions Instructions]"""
    elif topic_id == 'programs_courses':
        return """[Detailed Programs Instructions]"""
    elif topic_id == 'fees':
        return """[Detailed Fees Instructions]"""
    else:
        return """[Generic Instructions]"""
```

---

## Benefits

1. **Accurate Context Filtering**: LLM receives clear instructions on what to include/exclude
2. **Consistent Behavior**: Default behaviors clearly defined (e.g., default to new students)
3. **Program Matching**: Handles name variations and acronyms
4. **Student Type Separation**: Prevents mixing of different student type information
5. **Maintainable**: Easy to update instructions per topic without affecting others

---

## Future Enhancements

1. **Dynamic Program Detection**: Auto-detect program names/acronyms in queries
2. **Student Type Detection**: Pre-filter documents by detected student type
3. **Multi-Program Comparison**: Allow users to compare fees/requirements across programs
4. **Contextual Follow-ups**: Remember previous program/student type in conversation

---

## Notes

- These instructions are implemented through **prompt engineering** rather than code logic
- The LLM (Together AI) interprets these instructions when generating responses
- Document retrieval still uses the existing hybrid TF-IDF + Word2Vec system
- Topic keywords are defined in `backend/chatbot/topics.py`

---

## Maintenance

To update instructions for a topic:

1. Open `backend/chatbot/fast_hybrid_chatbot_together.py`
2. Find the `_get_topic_specific_instructions()` method
3. Modify the instructions for the specific topic_id
4. Test thoroughly with various query types

**Last Updated**: October 6, 2025

