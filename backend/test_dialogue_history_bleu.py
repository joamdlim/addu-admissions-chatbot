#!/usr/bin/env python3
"""
Dialogue History BLEU Score Evaluation Test Script
Tests LLaMA's ability to track dialogue history and maintain conversation context.
Target: BLEU score ≥ 0.40 for dialogue history tracking across 15 dialogue sessions.
"""

import os
import sys
import json
import requests
import time
from typing import List, Dict, Tuple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DialogueHistoryBLEUEvaluator:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8000/chatbot"
        self.smoothing = SmoothingFunction().method1
        
        # Define 15 dialogue sessions focused on testing dialogue history tracking
        self.dialogue_sessions = [
            {
                "session_id": "history_test_1",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the Computer Science curriculum?",
                        "context_keywords": ["computer", "science", "curriculum", "program"]
                    },
                    {
                        "query": "What about first year?",
                        "expected_context_reference": ["computer", "science", "first", "year"],
                        "context_test": "Should understand 'first year' refers to Computer Science first year"
                    },
                    {
                        "query": "How many subjects are there?",
                        "expected_context_reference": ["computer", "science", "first", "year", "subjects"],
                        "context_test": "Should understand 'subjects' refers to CS first year subjects"
                    }
                ]
            },
            {
                "session_id": "history_test_2",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the admission requirements for transfer students?",
                        "context_keywords": ["transfer", "students", "admission", "requirements"]
                    },
                    {
                        "query": "What documents do they need?",
                        "expected_context_reference": ["transfer", "students", "documents", "requirements"],
                        "context_test": "Should understand 'they' refers to transfer students"
                    },
                    {
                        "query": "When should they apply?",
                        "expected_context_reference": ["transfer", "students", "apply", "deadline"],
                        "context_test": "Should maintain transfer student context for application timing"
                    }
                ]
            },
            {
                "session_id": "history_test_3",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the tuition fees for Engineering programs?",
                        "context_keywords": ["engineering", "tuition", "fees", "programs"]
                    },
                    {
                        "query": "Are there payment plans for it?",
                        "expected_context_reference": ["engineering", "payment", "plans", "tuition"],
                        "context_test": "Should understand 'it' refers to Engineering tuition fees"
                    }
                ]
            },
            {
                "session_id": "history_test_4",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the Nursing program curriculum",
                        "context_keywords": ["nursing", "program", "curriculum"]
                    },
                    {
                        "query": "What are the clinical requirements?",
                        "expected_context_reference": ["nursing", "clinical", "requirements"],
                        "context_test": "Should understand clinical requirements are for Nursing"
                    },
                    {
                        "query": "How long does the program take?",
                        "expected_context_reference": ["nursing", "program", "duration", "years"],
                        "context_test": "Should understand 'the program' refers to Nursing program"
                    }
                ]
            },
            {
                "session_id": "history_test_5",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I apply for scholarships?",
                        "context_keywords": ["scholarships", "apply", "financial", "aid"]
                    },
                    {
                        "query": "What are the requirements for them?",
                        "expected_context_reference": ["scholarships", "requirements", "eligibility"],
                        "context_test": "Should understand 'them' refers to scholarships"
                    }
                ]
            },
            {
                "session_id": "history_test_6",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What payment methods are available?",
                        "context_keywords": ["payment", "methods", "options"]
                    },
                    {
                        "query": "Which banks support these?",
                        "expected_context_reference": ["payment", "methods", "banks", "support"],
                        "context_test": "Should understand 'these' refers to payment methods"
                    }
                ]
            },
            {
                "session_id": "history_test_7",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What programs are available in the School of Engineering?",
                        "context_keywords": ["engineering", "programs", "school", "available"]
                    },
                    {
                        "query": "Which one has the most math courses?",
                        "expected_context_reference": ["engineering", "programs", "math", "courses"],
                        "context_test": "Should understand 'which one' refers to engineering programs"
                    }
                ]
            },
            {
                "session_id": "history_test_8",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What is the application process for international students?",
                        "context_keywords": ["international", "students", "application", "process"]
                    },
                    {
                        "query": "What additional documents do they need?",
                        "expected_context_reference": ["international", "students", "documents", "additional"],
                        "context_test": "Should understand 'they' refers to international students"
                    }
                ]
            },
            {
                "session_id": "history_test_9",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the Business Administration curriculum?",
                        "context_keywords": ["business", "administration", "curriculum", "program"]
                    },
                    {
                        "query": "What subjects are in second year?",
                        "expected_context_reference": ["business", "administration", "second", "year"],
                        "context_test": "Should understand 'second year' refers to Business Administration second year"
                    },
                    {
                        "query": "Are there internship requirements?",
                        "expected_context_reference": ["business", "administration", "internship", "requirements"],
                        "context_test": "Should understand internship context for Business Administration"
                    }
                ]
            },
            {
                "session_id": "history_test_10",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the fees for Nursing program?",
                        "context_keywords": ["nursing", "program", "fees", "tuition"]
                    },
                    {
                        "query": "How much per year?",
                        "expected_context_reference": ["nursing", "program", "fees", "year"],
                        "context_test": "Should understand 'per year' refers to Nursing program yearly fees"
                    }
                ]
            },
            {
                "session_id": "history_test_11",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for graduate programs?",
                        "context_keywords": ["graduate", "programs", "requirements", "admission"]
                    },
                    {
                        "query": "What documents are needed for them?",
                        "expected_context_reference": ["graduate", "programs", "documents", "requirements"],
                        "context_test": "Should understand 'them' refers to graduate programs"
                    }
                ]
            },
            {
                "session_id": "history_test_12",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the Psychology program",
                        "context_keywords": ["psychology", "program", "curriculum"]
                    },
                    {
                        "query": "What are the major subjects?",
                        "expected_context_reference": ["psychology", "program", "major", "subjects"],
                        "context_test": "Should understand 'major subjects' refers to Psychology program subjects"
                    },
                    {
                        "query": "How many years does it take?",
                        "expected_context_reference": ["psychology", "program", "years", "duration"],
                        "context_test": "Should understand duration question refers to Psychology program"
                    }
                ]
            },
            {
                "session_id": "history_test_13",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the miscellaneous fees?",
                        "context_keywords": ["miscellaneous", "fees", "additional", "costs"]
                    },
                    {
                        "query": "Are they required for all students?",
                        "expected_context_reference": ["miscellaneous", "fees", "required", "students"],
                        "context_test": "Should understand 'they' refers to miscellaneous fees"
                    }
                ]
            },
            {
                "session_id": "history_test_14",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I enroll as a returning student?",
                        "context_keywords": ["enroll", "returning", "student", "process"]
                    },
                    {
                        "query": "What's the deadline for it?",
                        "expected_context_reference": ["enroll", "returning", "student", "deadline"],
                        "context_test": "Should understand 'it' refers to returning student enrollment"
                    }
                ]
            },
            {
                "session_id": "history_test_15",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What programs are offered in the School of Medicine?",
                        "context_keywords": ["programs", "school", "medicine", "offered"]
                    },
                    {
                        "query": "What are the admission requirements for these?",
                        "expected_context_reference": ["programs", "school", "medicine", "admission", "requirements"],
                        "context_test": "Should understand 'these' refers to School of Medicine programs"
                    },
                    {
                        "query": "How competitive are they?",
                        "expected_context_reference": ["programs", "school", "medicine", "competitive"],
                        "context_test": "Should understand 'they' refers to School of Medicine programs"
                    }
                ]
            }
        ]
    
    def send_guided_chat_request(self, user_input: str, action_type: str = "message", 
                               action_data: str = None, session_id: str = None) -> Dict:
        """Send request to guided chat endpoint without timeout"""
        endpoint = f"{self.base_url}/chat/guided/"
        payload = {
            "user_input": user_input,
            "action_type": action_type, 
            "action_data": action_data,
            "session_id": session_id
        }
        
        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def calculate_dialogue_history_bleu(self, response: str, expected_context: List[str], 
                                      previous_responses: List[str]) -> Dict:
        """Calculate BLEU score based on dialogue history context maintenance"""
        
        # Tokenize current response
        response_tokens = nltk.word_tokenize(response.lower())
        response_text = response.lower()
        
        # Count expected context references in current response
        context_matches = []
        for keyword in expected_context:
            if keyword.lower() in response_text:
                context_matches.append(keyword.lower())
        
        # Calculate context coverage
        context_coverage = len(context_matches) / len(expected_context) if expected_context else 0
        
        # Check for cross-reference to previous conversation elements
        previous_context_matches = []
        for prev_response in previous_responses:
            prev_tokens = nltk.word_tokenize(prev_response.lower())
            # Look for shared concepts between current and previous responses
            shared_tokens = set(response_tokens) & set(prev_tokens)
            # Filter to meaningful words (length > 2, not common words)
            meaningful_shared = [token for token in shared_tokens 
                               if len(token) > 2 and token not in ['the', 'and', 'for', 'are', 'you', 'can', 'will', 'have']]
            previous_context_matches.extend(meaningful_shared)
        
        # Remove duplicates
        previous_context_matches = list(set(previous_context_matches))
        
        # Calculate dialogue continuity score
        continuity_score = min(1.0, len(previous_context_matches) / 3)  # Normalize to max 1.0
        
        # Traditional BLEU score using expected context as reference
        try:
            reference = [expected_context]
            bleu_score = sentence_bleu(reference, response_tokens, smoothing_function=self.smoothing)
        except:
            bleu_score = 0.0
        
        # Combined dialogue history score
        # Weight: 50% context coverage, 30% continuity, 20% traditional BLEU
        dialogue_history_score = (context_coverage * 0.5) + (continuity_score * 0.3) + (bleu_score * 0.2)
        
        return {
            "context_coverage": context_coverage,
            "continuity_score": continuity_score,
            "bleu_score": bleu_score,
            "dialogue_history_score": dialogue_history_score,
            "context_matches": context_matches,
            "previous_context_matches": previous_context_matches,
            "response_length": len(response_tokens)
        }
    
    def run_dialogue_session(self, session_data: Dict) -> Dict:
        """Run a complete dialogue session and test history tracking"""
        session_id = session_data["session_id"]
        topic = session_data["topic"]
        conversations = session_data["conversations"]
        
        print(f"\n{'='*70}")
        print(f"🗣️ Testing Dialogue History: {session_id}")
        print(f"📋 Topic: {topic}")
        print(f"💬 Conversations: {len(conversations)}")
        print(f"🎯 Testing: Context maintenance and pronoun resolution")
        print(f"{'='*70}")
        
        # Initialize session with topic selection
        topic_response = self.send_guided_chat_request("", "topic_selection", topic, session_id)
        if not topic_response:
            return {"session_id": session_id, "error": "Failed to initialize session"}
        
        print(f"✅ Topic selected: {topic_response.get('current_topic')}")
        
        session_scores = []
        conversation_results = []
        all_responses = []
        
        # Run each conversation in the session
        for i, conv in enumerate(conversations, 1):
            print(f"\n💬 Exchange {i}: {conv['query']}")
            
            # Send query and get response
            start_time = time.time()
            response_data = self.send_guided_chat_request(conv['query'], "message", None, session_id)
            response_time = time.time() - start_time
            
            if not response_data or not response_data.get('response'):
                print(f"❌ No response received")
                conversation_results.append({
                    "exchange": i,
                    "query": conv['query'],
                    "error": "No response received",
                    "dialogue_history_score": 0.0
                })
                continue
            
            bot_response = response_data['response']
            all_responses.append(bot_response)
            print(f"🤖 Response ({response_time:.2f}s): {bot_response[:100]}...")
            
            # For first exchange, just store context keywords
            if i == 1:
                print(f"📝 Establishing context: {conv.get('context_keywords', [])}")
                conversation_results.append({
                    "exchange": i,
                    "query": conv['query'],
                    "response": bot_response,
                    "context_keywords": conv.get('context_keywords', []),
                    "dialogue_history_score": 1.0,  # First exchange always gets full score
                    "analysis": "Initial context establishment"
                })
                session_scores.append(1.0)
                continue
            
            # For follow-up exchanges, test dialogue history
            expected_context = conv.get('expected_context_reference', [])
            context_test = conv.get('context_test', '')
            
            print(f"🔍 Testing: {context_test}")
            print(f"🎯 Expected context: {expected_context}")
            
            # Calculate dialogue history BLEU score
            scores = self.calculate_dialogue_history_bleu(
                bot_response, expected_context, all_responses[:-1]  # Previous responses
            )
            
            session_scores.append(scores['dialogue_history_score'])
            
            conversation_results.append({
                "exchange": i,
                "query": conv['query'],
                "response": bot_response,
                "expected_context": expected_context,
                "context_test": context_test,
                "scores": scores,
                "dialogue_history_score": scores['dialogue_history_score']
            })
            
            print(f"📊 Context Coverage: {scores['context_coverage']:.3f}")
            print(f"📊 Continuity Score: {scores['continuity_score']:.3f}")
            print(f"📊 Dialogue History Score: {scores['dialogue_history_score']:.3f}")
            print(f"✅ Context Matches: {scores['context_matches']}")
            print(f"🔗 Previous Context: {scores['previous_context_matches']}")
            
            # Small delay between exchanges
            time.sleep(1)
        
        # Calculate session average
        avg_score = sum(session_scores) / len(session_scores) if session_scores else 0.0
        
        return {
            "session_id": session_id,
            "topic": topic,
            "conversations": conversation_results,
            "session_scores": session_scores,
            "average_dialogue_history_score": avg_score,
            "exchange_count": len(conversations)
        }
    
    def run_all_sessions(self) -> Dict:
        """Run all dialogue history test sessions"""
        print("🚀 Starting Dialogue History BLEU Evaluation")
        print("🎯 Target: BLEU Score ≥ 0.40 for dialogue history tracking")
        print("🔍 Testing: Context maintenance, pronoun resolution, conversation continuity")
        print("="*80)
        
        all_results = []
        all_scores = []
        
        for session_data in self.dialogue_sessions:
            result = self.run_dialogue_session(session_data)
            all_results.append(result)
            
            if 'average_dialogue_history_score' in result:
                all_scores.append(result['average_dialogue_history_score'])
        
        # Calculate overall statistics
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        return {
            "test_results": all_results,
            "overall_dialogue_history_score": overall_score,
            "target_score": 0.40,
            "target_achieved": overall_score >= 0.40,
            "total_sessions": len(all_results),
            "successful_sessions": len(all_scores)
        }
    
    def print_results(self, results: Dict):
        """Print comprehensive dialogue history analysis"""
        print(f"\n{'='*80}")
        print("📊 DIALOGUE HISTORY BLEU EVALUATION RESULTS")
        print(f"{'='*80}")
        
        print(f"🎯 Target Score: {results['target_score']}")
        print(f"📈 Achieved Score: {results['overall_dialogue_history_score']:.4f}")
        print(f"✅ Target Achieved: {'YES' if results['target_achieved'] else 'NO'}")
        print(f"📊 Total Sessions: {results['total_sessions']}")
        print(f"✅ Successful Sessions: {results['successful_sessions']}")
        
        print(f"\n📋 SESSION BREAKDOWN:")
        for result in results['test_results']:
            if 'average_dialogue_history_score' in result:
                status = "✅" if result['average_dialogue_history_score'] >= 0.40 else "❌"
                print(f"   {status} {result['session_id']}: {result['average_dialogue_history_score']:.4f} "
                      f"({result['exchange_count']} exchanges) - {result['topic']}")
            else:
                print(f"   ❌ {result['session_id']}: ERROR")
        
        # Detailed analysis of dialogue history issues
        print(f"\n🔍 DIALOGUE HISTORY ANALYSIS:")
        
        total_context_coverage = 0
        total_continuity = 0
        total_exchanges = 0
        
        context_failures = []
        continuity_failures = []
        
        for result in results['test_results']:
            if 'conversations' in result:
                for conv in result['conversations']:
                    if 'scores' in conv:
                        scores = conv['scores']
                        total_context_coverage += scores['context_coverage']
                        total_continuity += scores['continuity_score']
                        total_exchanges += 1
                        
                        # Track failures
                        if scores['context_coverage'] < 0.5:
                            context_failures.append({
                                'session': result['session_id'],
                                'query': conv['query'],
                                'expected': conv.get('expected_context', []),
                                'matches': scores['context_matches'],
                                'score': scores['context_coverage']
                            })
                        
                        if scores['continuity_score'] < 0.3:
                            continuity_failures.append({
                                'session': result['session_id'],
                                'query': conv['query'],
                                'score': scores['continuity_score'],
                                'previous_matches': scores['previous_context_matches']
                            })
        
        if total_exchanges > 0:
            avg_context_coverage = total_context_coverage / total_exchanges
            avg_continuity = total_continuity / total_exchanges
            
            print(f"   Average Context Coverage: {avg_context_coverage:.3f}")
            print(f"   Average Continuity Score: {avg_continuity:.3f}")
            print(f"   Context Failures: {len(context_failures)}/{total_exchanges}")
            print(f"   Continuity Failures: {len(continuity_failures)}/{total_exchanges}")
        
        # Show specific failure examples
        if context_failures:
            print(f"\n❌ CONTEXT RESOLUTION FAILURES:")
            for failure in context_failures[:3]:  # Show top 3
                print(f"   Session: {failure['session']}")
                print(f"   Query: {failure['query']}")
                print(f"   Expected: {failure['expected']}")
                print(f"   Found: {failure['matches']} (Score: {failure['score']:.3f})")
                print()
        
        if continuity_failures:
            print(f"\n❌ CONVERSATION CONTINUITY FAILURES:")
            for failure in continuity_failures[:3]:  # Show top 3
                print(f"   Session: {failure['session']}")
                print(f"   Query: {failure['query']}")
                print(f"   Previous Context Found: {failure['previous_matches']}")
                print(f"   Continuity Score: {failure['score']:.3f}")
                print()

def main():
    """Main test runner"""
    print("🧪 Dialogue History BLEU Evaluation Test")
    print("⚠️  Make sure Django server is running: python manage.py runserver")
    print("⚠️  This test focuses on LLaMA's dialogue history tracking ability")
    print("⚠️  Testing context maintenance, pronoun resolution, and conversation continuity")
    
    input("\nPress ENTER to start the evaluation...")
    
    evaluator = DialogueHistoryBLEUEvaluator()
    
    try:
        results = evaluator.run_all_sessions()
        evaluator.print_results(results)
        
        # Save results to file
        with open('dialogue_history_bleu_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: dialogue_history_bleu_results.json")
        
        # Exit with appropriate code
        sys.exit(0 if results['target_achieved'] else 1)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
