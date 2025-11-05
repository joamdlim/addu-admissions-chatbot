#!/usr/bin/env python3
"""
F1-Score Evaluation Test Script for Guided Conversations
Tests fuzzy typo correction and next-word prediction accuracy using guided chatbot.
Target: F1-score ≥ 0.70 across 30 input samples.
"""

import os
import sys
import json
import time
import requests
from typing import List, Dict, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score
import difflib

# Setup Django environment for direct function access
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

try:
    import django
    django.setup()
    from chatbot.together_ai_interface import correct_typos, predict_next_words
    DIRECT_ACCESS = True
except Exception as e:
    print(f"⚠️  Direct function access failed: {e}")
    print("Will test through guided conversation interface only")
    DIRECT_ACCESS = False

class F1ScoreEvaluatorGuided:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8000/chatbot"
        
        # 30 test samples for typo correction (15) and next-word prediction (15)
        self.typo_correction_samples = [
            # Basic typos in admission context
            {"input": "admision requirements", "expected": "admission requirements"},
            {"input": "tution fees", "expected": "tuition fees"},
            {"input": "aplication deadline", "expected": "application deadline"},
            {"input": "scholarhip information", "expected": "scholarship information"},
            {"input": "enrollmnt process", "expected": "enrollment process"},
            
            # Program-related typos
            {"input": "comuter science progam", "expected": "computer science program"},
            {"input": "enginerring cours", "expected": "engineering course"},
            {"input": "nursng curriculum", "expected": "nursing curriculum"},
            {"input": "busines program", "expected": "business program"},
            {"input": "informaton technology", "expected": "information technology"},
            
            # Fee-related typos
            {"input": "paymet plans", "expected": "payment plans"},
            {"input": "finacial aid", "expected": "financial aid"},
            {"input": "semster fees", "expected": "semester fees"},
            {"input": "anual cost", "expected": "annual cost"},
            {"input": "installement options", "expected": "installment options"}
        ]
        
        # 15 test samples for next-word prediction in guided context
        self.word_prediction_samples = [
            # Admission context predictions
            {"input": "admission", "expected_words": ["requirements", "process", "deadline", "office", "application"]},
            {"input": "scholarship", "expected_words": ["information", "requirements", "application", "eligibility", "program"]},
            {"input": "financial", "expected_words": ["aid", "assistance", "support", "help", "options"]},
            {"input": "transfer", "expected_words": ["student", "credits", "evaluation", "requirements", "process"]},
            {"input": "international", "expected_words": ["student", "requirements", "admission", "visa", "documents"]},
            
            # Program context predictions
            {"input": "computer science", "expected_words": ["curriculum", "program", "course", "degree", "subjects"]},
            {"input": "engineering", "expected_words": ["curriculum", "program", "course", "degree", "subjects"]},
            {"input": "nursing", "expected_words": ["curriculum", "program", "course", "degree", "subjects"]},
            {"input": "business", "expected_words": ["curriculum", "program", "course", "degree", "subjects"]},
            {"input": "first year", "expected_words": ["courses", "subjects", "curriculum", "program", "schedule"]},
            
            # Fee context predictions
            {"input": "tuition", "expected_words": ["fees", "cost", "payment", "price", "amount"]},
            {"input": "payment", "expected_words": ["plan", "method", "schedule", "options", "deadline"]},
            {"input": "semester", "expected_words": ["fees", "cost", "payment", "tuition", "breakdown"]},
            {"input": "annual", "expected_words": ["cost", "fees", "tuition", "payment", "amount"]},
            {"input": "installment", "expected_words": ["plan", "payment", "options", "schedule", "terms"]}
        ]
        
        self.typo_results = []
        self.prediction_results = []
    
    def send_guided_chat_request(self, user_input: str, action_type: str = "message", 
                               action_data: str = None, session_id: str = None) -> Dict:
        """Send request to guided chat endpoint"""
        endpoint = f"{self.base_url}/chat/guided/"
        payload = {
            "user_input": user_input,
            "action_type": action_type, 
            "action_data": action_data,
            "session_id": session_id or "f1_test_session"
        }
        
        try:
            # Remove timeout to handle slow responses
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def evaluate_typo_correction_direct(self) -> Dict:
        """Evaluate typo correction using direct function access"""
        print("🔤 Evaluating Typo Correction (Direct Access)...")
        print("="*50)
        
        correct_predictions = 0
        total_predictions = len(self.typo_correction_samples)
        
        for i, sample in enumerate(self.typo_correction_samples, 1):
            input_text = sample["input"]
            expected = sample["expected"]
            
            print(f"\n{i:2d}. Input: '{input_text}'")
            print(f"    Expected: '{expected}'")
            
            try:
                # Test typo correction
                corrected = correct_typos(input_text)
                print(f"    Corrected: '{corrected}'")
                
                # Calculate similarity score
                similarity = difflib.SequenceMatcher(None, corrected.lower(), expected.lower()).ratio()
                
                # Consider it correct if similarity >= 0.8
                is_correct = similarity >= 0.8
                if is_correct:
                    correct_predictions += 1
                
                result = {
                    "input": input_text,
                    "expected": expected,
                    "corrected": corrected,
                    "similarity": similarity,
                    "correct": is_correct
                }
                
                self.typo_results.append(result)
                
                status = "✅" if is_correct else "❌"
                print(f"    {status} Similarity: {similarity:.3f}")
                
                time.sleep(0.1)  # Small delay to avoid rate limiting
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                self.typo_results.append({
                    "input": input_text,
                    "expected": expected,
                    "error": str(e),
                    "correct": False
                })
        
        accuracy = correct_predictions / total_predictions
        
        return {
            "total_samples": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "results": self.typo_results
        }
    
    def evaluate_typo_correction_guided(self) -> Dict:
        """Evaluate typo correction through guided conversation"""
        print("🔤 Evaluating Typo Correction (Guided Conversation)...")
        print("="*50)
        
        # Initialize session with admissions topic
        session_id = "typo_test_session"
        topic_response = self.send_guided_chat_request("", "topic_selection", "admissions_enrollment", session_id)
        
        if not topic_response:
            print("❌ Failed to initialize guided session")
            return {"error": "Failed to initialize session"}
        
        correct_predictions = 0
        total_predictions = len(self.typo_correction_samples)
        
        for i, sample in enumerate(self.typo_correction_samples, 1):
            input_text = sample["input"]
            expected = sample["expected"]
            
            print(f"\n{i:2d}. Input: '{input_text}'")
            print(f"    Expected: '{expected}'")
            
            try:
                # Send typo-laden query to guided chat
                response_data = self.send_guided_chat_request(input_text, "message", None, session_id)
                
                if response_data and response_data.get('response'):
                    bot_response = response_data['response']
                    
                    # Check if the bot understood the query correctly despite typos
                    # by looking for key terms from expected correction in the response
                    expected_terms = expected.lower().split()
                    response_lower = bot_response.lower()
                    
                    # Count how many expected terms appear in the response
                    terms_found = sum(1 for term in expected_terms if term in response_lower)
                    understanding_score = terms_found / len(expected_terms)
                    
                    # Consider it correct if bot understood >= 70% of the corrected terms
                    is_correct = understanding_score >= 0.7
                    if is_correct:
                        correct_predictions += 1
                    
                    result = {
                        "input": input_text,
                        "expected": expected,
                        "bot_response": bot_response[:100] + "...",
                        "understanding_score": understanding_score,
                        "correct": is_correct
                    }
                    
                    self.typo_results.append(result)
                    
                    status = "✅" if is_correct else "❌"
                    print(f"    {status} Understanding: {understanding_score:.3f}")
                    
                else:
                    print(f"    ❌ No response received")
                    self.typo_results.append({
                        "input": input_text,
                        "expected": expected,
                        "error": "No response",
                        "correct": False
                    })
                
                time.sleep(0.5)  # Delay between requests
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                self.typo_results.append({
                    "input": input_text,
                    "expected": expected,
                    "error": str(e),
                    "correct": False
                })
        
        accuracy = correct_predictions / total_predictions
        
        return {
            "total_samples": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "results": self.typo_results
        }
    
    def evaluate_word_prediction_direct(self) -> Dict:
        """Evaluate next-word prediction using direct function access"""
        print("\n🔮 Evaluating Next-Word Prediction (Direct Access)...")
        print("="*50)
        
        correct_predictions = 0
        total_predictions = len(self.word_prediction_samples)
        
        for i, sample in enumerate(self.word_prediction_samples, 1):
            input_text = sample["input"]
            expected_words = sample["expected_words"]
            
            print(f"\n{i:2d}. Input: '{input_text}'")
            print(f"    Expected: {expected_words}")
            
            try:
                # Test word prediction
                predictions = predict_next_words(input_text, num_suggestions=3)
                print(f"    Predicted: {predictions}")
                
                # Check if any prediction matches expected words
                is_correct = False
                for prediction in predictions:
                    prediction_words = prediction.lower().split()
                    for word in prediction_words:
                        if word in [w.lower() for w in expected_words]:
                            is_correct = True
                            break
                    if is_correct:
                        break
                
                if is_correct:
                    correct_predictions += 1
                
                result = {
                    "input": input_text,
                    "expected_words": expected_words,
                    "predictions": predictions,
                    "correct": is_correct
                }
                
                self.prediction_results.append(result)
                
                status = "✅" if is_correct else "❌"
                print(f"    {status} Match found: {is_correct}")
                
                time.sleep(0.1)  # Small delay to avoid rate limiting
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                self.prediction_results.append({
                    "input": input_text,
                    "expected_words": expected_words,
                    "error": str(e),
                    "correct": False
                })
        
        accuracy = correct_predictions / total_predictions
        
        return {
            "total_samples": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "results": self.prediction_results
        }
    
    def evaluate_word_prediction_guided(self) -> Dict:
        """Evaluate contextual word prediction through guided conversation"""
        print("\n🔮 Evaluating Contextual Word Prediction (Guided Conversation)...")
        print("="*50)
        
        correct_predictions = 0
        total_predictions = len(self.word_prediction_samples)
        
        # Test across different topics for context variety
        topics = ["admissions_enrollment", "programs_courses", "fees"]
        
        for i, sample in enumerate(self.word_prediction_samples, 1):
            input_text = sample["input"]
            expected_words = sample["expected_words"]
            
            # Select appropriate topic based on input context
            if any(word in input_text.lower() for word in ["admission", "scholarship", "financial", "transfer", "international"]):
                topic = "admissions_enrollment"
            elif any(word in input_text.lower() for word in ["computer", "engineering", "nursing", "business", "year", "curriculum"]):
                topic = "programs_courses"
            else:
                topic = "fees"
            
            session_id = f"prediction_test_{i}"
            
            print(f"\n{i:2d}. Input: '{input_text}' (Topic: {topic})")
            print(f"    Expected: {expected_words}")
            
            try:
                # Initialize session with appropriate topic
                topic_response = self.send_guided_chat_request("", "topic_selection", topic, session_id)
                
                if not topic_response:
                    print(f"    ❌ Failed to initialize session")
                    continue
                
                # Send incomplete query to see if bot provides contextual completion
                incomplete_query = f"Tell me about {input_text}"
                response_data = self.send_guided_chat_request(incomplete_query, "message", None, session_id)
                
                if response_data and response_data.get('response'):
                    bot_response = response_data['response']
                    
                    # Check if response contains expected contextual words
                    response_lower = bot_response.lower()
                    matches_found = []
                    
                    for expected_word in expected_words:
                        if expected_word.lower() in response_lower:
                            matches_found.append(expected_word)
                    
                    # Consider it correct if at least one expected word appears
                    is_correct = len(matches_found) > 0
                    if is_correct:
                        correct_predictions += 1
                    
                    result = {
                        "input": input_text,
                        "topic": topic,
                        "expected_words": expected_words,
                        "bot_response": bot_response[:100] + "...",
                        "matches_found": matches_found,
                        "correct": is_correct
                    }
                    
                    self.prediction_results.append(result)
                    
                    status = "✅" if is_correct else "❌"
                    print(f"    {status} Matches: {matches_found}")
                    
                else:
                    print(f"    ❌ No response received")
                    self.prediction_results.append({
                        "input": input_text,
                        "expected_words": expected_words,
                        "error": "No response",
                        "correct": False
                    })
                
                time.sleep(0.5)  # Delay between requests
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                self.prediction_results.append({
                    "input": input_text,
                    "expected_words": expected_words,
                    "error": str(e),
                    "correct": False
                })
        
        accuracy = correct_predictions / total_predictions
        
        return {
            "total_samples": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "results": self.prediction_results
        }
    
    def calculate_f1_score(self, typo_results: Dict, prediction_results: Dict) -> Dict:
        """Calculate combined F1 score"""
        # Combine results for F1 calculation
        all_correct = []
        all_predicted = []
        
        # Add typo correction results
        for result in typo_results["results"]:
            all_correct.append(1)  # Ground truth (all should be handled correctly)
            all_predicted.append(1 if result.get("correct", False) else 0)
        
        # Add word prediction results
        for result in prediction_results["results"]:
            all_correct.append(1)  # Ground truth (all should be predicted correctly)
            all_predicted.append(1 if result.get("correct", False) else 0)
        
        # Calculate metrics
        if len(all_correct) > 0:
            precision = precision_score(all_correct, all_predicted, zero_division=0)
            recall = recall_score(all_correct, all_predicted, zero_division=0)
            f1 = f1_score(all_correct, all_predicted, zero_division=0)
        else:
            precision = recall = f1 = 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "total_samples": len(all_correct),
            "correct_predictions": sum(all_predicted),
            "typo_accuracy": typo_results["accuracy"],
            "prediction_accuracy": prediction_results["accuracy"]
        }
    
    def run_evaluation(self) -> Dict:
        """Run complete F1 evaluation"""
        print("🚀 Starting F1-Score Evaluation for Guided Conversations")
        print("🎯 Target: F1-Score ≥ 0.70 across 30 input samples")
        print("="*80)
        
        # Choose evaluation method based on availability
        if DIRECT_ACCESS:
            print("Using direct function access for more accurate testing")
            typo_results = self.evaluate_typo_correction_direct()
            prediction_results = self.evaluate_word_prediction_direct()
        else:
            print("Using guided conversation interface for testing")
            typo_results = self.evaluate_typo_correction_guided()
            prediction_results = self.evaluate_word_prediction_guided()
        
        # Calculate combined F1 score
        f1_results = self.calculate_f1_score(typo_results, prediction_results)
        
        return {
            "typo_correction": typo_results,
            "word_prediction": prediction_results,
            "f1_metrics": f1_results,
            "target_f1": 0.70,
            "target_achieved": f1_results["f1_score"] >= 0.70
        }
    
    def print_results(self, results: Dict):
        """Print comprehensive results"""
        print(f"\n{'='*80}")
        print("📊 F1-SCORE EVALUATION RESULTS")
        print(f"{'='*80}")
        
        f1_metrics = results["f1_metrics"]
        
        print(f"🎯 Target F1-Score: {results['target_f1']}")
        print(f"📈 Achieved F1-Score: {f1_metrics['f1_score']:.4f}")
        print(f"✅ Target Achieved: {'YES' if results['target_achieved'] else 'NO'}")
        print(f"📊 Total Samples: {f1_metrics['total_samples']}")
        print(f"✅ Correct Predictions: {f1_metrics['correct_predictions']}")
        
        print(f"\n📋 DETAILED METRICS:")
        print(f"   Precision: {f1_metrics['precision']:.4f}")
        print(f"   Recall: {f1_metrics['recall']:.4f}")
        print(f"   F1-Score: {f1_metrics['f1_score']:.4f}")
        
        print(f"\n📝 COMPONENT BREAKDOWN:")
        print(f"   Typo Correction Accuracy: {f1_metrics['typo_accuracy']:.4f}")
        print(f"   Word Prediction Accuracy: {f1_metrics['prediction_accuracy']:.4f}")

def main():
    """Main test runner"""
    print("🧪 F1-Score Evaluation Test for Guided Conversations")
    print("⚠️  Make sure Django server is running: python manage.py runserver")
    print("⚠️  This test evaluates typo correction and next-word prediction")
    print("⚠️  No timeouts - responses may take time to generate")
    
    input("\nPress ENTER to start the evaluation...")
    
    evaluator = F1ScoreEvaluatorGuided()
    
    try:
        results = evaluator.run_evaluation()
        evaluator.print_results(results)
        
        # Save results to file
        with open('f1_evaluation_guided_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: f1_evaluation_guided_results.json")
        
        # Exit with appropriate code
        sys.exit(0 if results['target_achieved'] else 1)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
