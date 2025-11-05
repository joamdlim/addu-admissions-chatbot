#!/usr/bin/env python3
"""
Hybrid Retrieval System Semantic Relevance Evaluation
Tests the existing TF-IDF + Word2Vec hybrid retrieval system in the guided chatbot.
Target: F1-score ≥ 0.70 on semantic relevance tests over 50 varied user queries.
"""

import os
import sys
import json
import time
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support
import requests

# Add the current directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class HybridRetrievalSemanticEvaluator:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8000/chatbot"
        
        # Define 50 varied user queries with expected relevant topics/documents
        self.test_queries = [
            # Admissions & Enrollment Queries (15 queries)
            {
                "query": "What are the admission requirements for first-year students?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["admission", "requirements", "first-year", "students", "documents"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "How do I apply for scholarships and financial aid?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["scholarships", "financial", "aid", "apply", "assistance"],
                "semantic_category": "financial_assistance"
            },
            {
                "query": "What documents do transfer students need to submit?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["transfer", "students", "documents", "submit", "requirements"],
                "semantic_category": "transfer_requirements"
            },
            {
                "query": "When is the application deadline for international students?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["application", "deadline", "international", "students"],
                "semantic_category": "international_admission"
            },
            {
                "query": "How do I enroll as a returning student?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["enroll", "returning", "student", "process"],
                "semantic_category": "returning_student"
            },
            {
                "query": "What are the eligibility criteria for merit scholarships?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["eligibility", "criteria", "merit", "scholarships"],
                "semantic_category": "scholarship_eligibility"
            },
            {
                "query": "Can I get financial assistance for my studies?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["financial", "assistance", "studies", "aid"],
                "semantic_category": "financial_assistance"
            },
            {
                "query": "What is the process for graduate school admission?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["process", "graduate", "school", "admission"],
                "semantic_category": "graduate_admission"
            },
            {
                "query": "How do I submit my high school transcripts?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["submit", "high", "school", "transcripts", "documents"],
                "semantic_category": "document_submission"
            },
            {
                "query": "What are the requirements for student visa application?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["requirements", "student", "visa", "application", "international"],
                "semantic_category": "visa_requirements"
            },
            {
                "query": "Is there need-based financial aid available?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["need-based", "financial", "aid", "available"],
                "semantic_category": "financial_assistance"
            },
            {
                "query": "How do I check my admission status?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["check", "admission", "status", "application"],
                "semantic_category": "admission_status"
            },
            {
                "query": "What are the English proficiency requirements?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["English", "proficiency", "requirements", "international"],
                "semantic_category": "language_requirements"
            },
            {
                "query": "Can I defer my admission to next semester?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["defer", "admission", "next", "semester"],
                "semantic_category": "admission_deferral"
            },
            {
                "query": "What is the minimum GPA requirement for admission?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["minimum", "GPA", "requirement", "admission"],
                "semantic_category": "academic_requirements"
            },
            
            # Programs & Courses Queries (20 queries)
            {
                "query": "What is the Computer Science curriculum?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["computer", "science", "curriculum", "program"],
                "semantic_category": "program_curriculum"
            },
            {
                "query": "Tell me about the Engineering programs offered",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["engineering", "programs", "offered"],
                "semantic_category": "program_information"
            },
            {
                "query": "What subjects are taught in first year Nursing?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["subjects", "first", "year", "nursing", "curriculum"],
                "semantic_category": "program_curriculum"
            },
            {
                "query": "How many years is the Business Administration program?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["years", "business", "administration", "program", "duration"],
                "semantic_category": "program_duration"
            },
            {
                "query": "What are the major subjects in Psychology?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["major", "subjects", "psychology", "curriculum"],
                "semantic_category": "program_curriculum"
            },
            {
                "query": "Is there an Information Technology degree program?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["information", "technology", "degree", "program"],
                "semantic_category": "program_availability"
            },
            {
                "query": "What are the clinical requirements for Nursing students?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["clinical", "requirements", "nursing", "students"],
                "semantic_category": "program_requirements"
            },
            {
                "query": "Which programs are available in the School of Medicine?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["programs", "available", "school", "medicine"],
                "semantic_category": "program_information"
            },
            {
                "query": "What mathematics courses are required for Engineering?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["mathematics", "courses", "required", "engineering"],
                "semantic_category": "program_requirements"
            },
            {
                "query": "How many units is the Computer Science program?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["units", "computer", "science", "program"],
                "semantic_category": "program_details"
            },
            {
                "query": "What are the internship requirements for Business students?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["internship", "requirements", "business", "students"],
                "semantic_category": "program_requirements"
            },
            {
                "query": "Are there online courses available?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["online", "courses", "available"],
                "semantic_category": "course_delivery"
            },
            {
                "query": "What is the curriculum for second year IT students?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["curriculum", "second", "year", "IT", "students"],
                "semantic_category": "program_curriculum"
            },
            {
                "query": "Which programs have the most laboratory work?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["programs", "laboratory", "work"],
                "semantic_category": "program_characteristics"
            },
            {
                "query": "What are the thesis requirements for graduating students?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["thesis", "requirements", "graduating", "students"],
                "semantic_category": "graduation_requirements"
            },
            {
                "query": "Are there summer classes for Engineering programs?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["summer", "classes", "engineering", "programs"],
                "semantic_category": "course_scheduling"
            },
            {
                "query": "What programming languages are taught in Computer Science?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["programming", "languages", "taught", "computer", "science"],
                "semantic_category": "program_curriculum"
            },
            {
                "query": "How many semesters is the Nursing program?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["semesters", "nursing", "program", "duration"],
                "semantic_category": "program_duration"
            },
            {
                "query": "What are the prerequisites for advanced courses?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["prerequisites", "advanced", "courses"],
                "semantic_category": "course_requirements"
            },
            {
                "query": "Which degree programs offer double major options?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["degree", "programs", "double", "major", "options"],
                "semantic_category": "program_options"
            },
            
            # Fees & Payment Queries (15 queries)
            {
                "query": "What are the tuition fees for Engineering programs?",
                "expected_topics": ["fees"],
                "expected_keywords": ["tuition", "fees", "engineering", "programs"],
                "semantic_category": "program_fees"
            },
            {
                "query": "How much does it cost to study Computer Science?",
                "expected_topics": ["fees"],
                "expected_keywords": ["cost", "study", "computer", "science", "fees"],
                "semantic_category": "program_fees"
            },
            {
                "query": "Are there payment plans available for tuition?",
                "expected_topics": ["fees"],
                "expected_keywords": ["payment", "plans", "available", "tuition"],
                "semantic_category": "payment_options"
            },
            {
                "query": "What are the miscellaneous fees for students?",
                "expected_topics": ["fees"],
                "expected_keywords": ["miscellaneous", "fees", "students"],
                "semantic_category": "additional_fees"
            },
            {
                "query": "How much are the laboratory fees for Nursing?",
                "expected_topics": ["fees"],
                "expected_keywords": ["laboratory", "fees", "nursing"],
                "semantic_category": "specific_fees"
            },
            {
                "query": "What payment methods are accepted by the university?",
                "expected_topics": ["fees"],
                "expected_keywords": ["payment", "methods", "accepted", "university"],
                "semantic_category": "payment_methods"
            },
            {
                "query": "Is there a discount for early payment of fees?",
                "expected_topics": ["fees"],
                "expected_keywords": ["discount", "early", "payment", "fees"],
                "semantic_category": "payment_discounts"
            },
            {
                "query": "What is the total cost for a 4-year program?",
                "expected_topics": ["fees"],
                "expected_keywords": ["total", "cost", "4-year", "program"],
                "semantic_category": "total_program_cost"
            },
            {
                "query": "Are there additional fees for international students?",
                "expected_topics": ["fees"],
                "expected_keywords": ["additional", "fees", "international", "students"],
                "semantic_category": "international_fees"
            },
            {
                "query": "How much is the enrollment fee?",
                "expected_topics": ["fees"],
                "expected_keywords": ["enrollment", "fee", "cost"],
                "semantic_category": "enrollment_fees"
            },
            {
                "query": "What are the fees for summer classes?",
                "expected_topics": ["fees"],
                "expected_keywords": ["fees", "summer", "classes"],
                "semantic_category": "summer_fees"
            },
            {
                "query": "Can I pay my tuition in installments?",
                "expected_topics": ["fees"],
                "expected_keywords": ["pay", "tuition", "installments"],
                "semantic_category": "payment_options"
            },
            {
                "query": "What is the refund policy for tuition fees?",
                "expected_topics": ["fees"],
                "expected_keywords": ["refund", "policy", "tuition", "fees"],
                "semantic_category": "refund_policy"
            },
            {
                "query": "Are there any hidden costs in the program fees?",
                "expected_topics": ["fees"],
                "expected_keywords": ["hidden", "costs", "program", "fees"],
                "semantic_category": "fee_transparency"
            },
            {
                "query": "What banks support online payment for fees?",
                "expected_topics": ["fees"],
                "expected_keywords": ["banks", "support", "online", "payment", "fees"],
                "semantic_category": "payment_methods"
            }
        ]
    
    def send_guided_chat_request(self, user_input: str, action_type: str = "message", 
                               action_data: str = None, session_id: str = None) -> Dict:
        """Send request to guided chat endpoint"""
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
    
    def evaluate_semantic_relevance(self, query_data: Dict, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate semantic relevance of retrieved documents"""
        expected_topics = query_data["expected_topics"]
        expected_keywords = query_data["expected_keywords"]
        semantic_category = query_data["semantic_category"]
        
        # Check topic relevance
        topic_matches = 0
        for doc in retrieved_docs:
            doc_topic = doc.get('current_topic', '')
            if doc_topic in expected_topics:
                topic_matches += 1
        
        topic_relevance = topic_matches / len(expected_topics) if expected_topics else 0
        
        # Check keyword relevance
        keyword_matches = 0
        total_content = ""
        for doc in retrieved_docs:
            content = doc.get('content', '').lower()
            total_content += " " + content
        
        for keyword in expected_keywords:
            if keyword.lower() in total_content:
                keyword_matches += 1
        
        keyword_relevance = keyword_matches / len(expected_keywords) if expected_keywords else 0
        
        # Check retrieval strategy (should use hybrid)
        hybrid_used = any(
            'hybrid' in doc.get('retrieval_strategy', '').lower() or
            doc.get('_debug', {}).get('tfidf_word2vec_used', False)
            for doc in retrieved_docs
        )
        
        # Calculate semantic scores from debug info if available
        semantic_scores = []
        tfidf_scores = []
        for doc in retrieved_docs:
            debug_info = doc.get('_debug', {})
            if 'semantic_similarity' in debug_info:
                semantic_scores.append(debug_info['semantic_similarity'])
            if 'specialized_score' in debug_info:
                tfidf_scores.append(debug_info['specialized_score'])
        
        avg_semantic_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0
        avg_tfidf_score = sum(tfidf_scores) / len(tfidf_scores) if tfidf_scores else 0
        
        # Overall relevance score (weighted combination)
        overall_relevance = (topic_relevance * 0.4) + (keyword_relevance * 0.4) + (avg_semantic_score * 0.2)
        
        # Binary classification: relevant if overall score >= 0.5
        is_relevant = overall_relevance >= 0.5
        
        return {
            "is_relevant": is_relevant,
            "overall_relevance": overall_relevance,
            "topic_relevance": topic_relevance,
            "keyword_relevance": keyword_relevance,
            "avg_semantic_score": avg_semantic_score,
            "avg_tfidf_score": avg_tfidf_score,
            "hybrid_used": hybrid_used,
            "topic_matches": topic_matches,
            "keyword_matches": keyword_matches,
            "semantic_category": semantic_category,
            "retrieved_count": len(retrieved_docs)
        }
    
    def run_semantic_evaluation(self) -> Dict:
        """Run semantic relevance evaluation on all test queries"""
        print("🚀 Starting Hybrid Retrieval Semantic Relevance Evaluation")
        print("🎯 Target: F1-Score ≥ 0.70 on semantic relevance tests")
        print("🔍 Testing: TF-IDF + Word2Vec hybrid retrieval system")
        print(f"📊 Test Queries: {len(self.test_queries)}")
        print("="*80)
        
        results = []
        y_true = []  # Ground truth (all should be relevant)
        y_pred = []  # Predictions from retrieval system
        
        category_results = {}
        
        for i, query_data in enumerate(self.test_queries, 1):
            query = query_data["query"]
            expected_topics = query_data["expected_topics"]
            semantic_category = query_data["semantic_category"]
            
            print(f"\n📝 Query {i}/50: {query}")
            print(f"🎯 Expected Topics: {expected_topics}")
            print(f"🏷️ Category: {semantic_category}")
            
            # Initialize session with appropriate topic
            topic_to_test = expected_topics[0] if expected_topics else "admissions_enrollment"
            session_id = f"semantic_test_{i}"
            
            # Select topic
            topic_response = self.send_guided_chat_request("", "topic_selection", topic_to_test, session_id)
            if not topic_response:
                print(f"❌ Failed to initialize session for query {i}")
                continue
            
            print(f"✅ Topic selected: {topic_response.get('current_topic')}")
            
            # Send query and get retrieval results
            start_time = time.time()
            response_data = self.send_guided_chat_request(query, "message", None, session_id)
            response_time = time.time() - start_time
            
            if not response_data:
                print(f"❌ No response for query {i}")
                y_true.append(1)  # Expected to be relevant
                y_pred.append(0)  # Predicted as not relevant
                continue
            
            # Extract retrieval information
            retrieved_docs = response_data.get('sources', [])
            print(f"📚 Retrieved {len(retrieved_docs)} documents ({response_time:.2f}s)")
            
            # Evaluate semantic relevance
            evaluation = self.evaluate_semantic_relevance(query_data, retrieved_docs)
            
            # Store results
            result = {
                "query_id": i,
                "query": query,
                "expected_topics": expected_topics,
                "semantic_category": semantic_category,
                "evaluation": evaluation,
                "retrieved_docs": len(retrieved_docs),
                "response_time": response_time
            }
            results.append(result)
            
            # For F1 calculation
            y_true.append(1)  # All queries should retrieve relevant documents
            y_pred.append(1 if evaluation["is_relevant"] else 0)
            
            # Category tracking
            if semantic_category not in category_results:
                category_results[semantic_category] = {"correct": 0, "total": 0}
            category_results[semantic_category]["total"] += 1
            if evaluation["is_relevant"]:
                category_results[semantic_category]["correct"] += 1
            
            # Print evaluation results
            status = "✅" if evaluation["is_relevant"] else "❌"
            print(f"{status} Relevance: {evaluation['overall_relevance']:.3f}")
            print(f"   Topic Match: {evaluation['topic_relevance']:.3f}")
            print(f"   Keyword Match: {evaluation['keyword_relevance']:.3f}")
            print(f"   Semantic Score: {evaluation['avg_semantic_score']:.3f}")
            print(f"   Hybrid Used: {evaluation['hybrid_used']}")
            
            # Small delay between queries
            time.sleep(0.5)
        
        # Calculate overall metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        
        return {
            "results": results,
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
                "target_f1": 0.70,
                "target_achieved": f1 >= 0.70
            },
            "category_results": category_results,
            "total_queries": len(self.test_queries),
            "relevant_predictions": sum(y_pred),
            "y_true": y_true,
            "y_pred": y_pred
        }
    
    def print_results(self, results: Dict):
        """Print comprehensive evaluation results"""
        print(f"\n{'='*80}")
        print("📊 HYBRID RETRIEVAL SEMANTIC RELEVANCE EVALUATION RESULTS")
        print(f"{'='*80}")
        
        metrics = results["metrics"]
        print(f"🎯 Target F1-Score: {metrics['target_f1']}")
        print(f"📈 Achieved F1-Score: {metrics['f1_score']:.4f}")
        print(f"✅ Target Achieved: {'YES' if metrics['target_achieved'] else 'NO'}")
        print(f"📊 Total Queries: {results['total_queries']}")
        print(f"✅ Relevant Predictions: {results['relevant_predictions']}")
        
        print(f"\n📋 DETAILED METRICS:")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        
        # Category breakdown
        print(f"\n📝 SEMANTIC CATEGORY PERFORMANCE:")
        category_results = results["category_results"]
        for category, stats in category_results.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            status = "✅" if accuracy >= 0.7 else "❌"
            print(f"   {status} {category}: {accuracy:.3f} ({stats['correct']}/{stats['total']})")
        
        # Hybrid system analysis
        print(f"\n🔍 HYBRID SYSTEM ANALYSIS:")
        hybrid_usage = sum(1 for r in results["results"] if r["evaluation"]["hybrid_used"])
        print(f"   Hybrid Retrieval Used: {hybrid_usage}/{results['total_queries']} queries")
        
        avg_semantic_score = sum(r["evaluation"]["avg_semantic_score"] for r in results["results"]) / len(results["results"])
        avg_tfidf_score = sum(r["evaluation"]["avg_tfidf_score"] for r in results["results"]) / len(results["results"])
        
        print(f"   Average Semantic (Word2Vec) Score: {avg_semantic_score:.3f}")
        print(f"   Average TF-IDF Score: {avg_tfidf_score:.3f}")
        
        # Performance by topic
        print(f"\n📋 PERFORMANCE BY TOPIC:")
        topic_stats = {}
        for result in results["results"]:
            for topic in result["expected_topics"]:
                if topic not in topic_stats:
                    topic_stats[topic] = {"correct": 0, "total": 0}
                topic_stats[topic]["total"] += 1
                if result["evaluation"]["is_relevant"]:
                    topic_stats[topic]["correct"] += 1
        
        for topic, stats in topic_stats.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            status = "✅" if accuracy >= 0.7 else "❌"
            print(f"   {status} {topic}: {accuracy:.3f} ({stats['correct']}/{stats['total']})")

def main():
    """Main test runner"""
    print("🧪 Hybrid Retrieval System Semantic Relevance Evaluation")
    print("⚠️  Make sure Django server is running: python manage.py runserver")
    print("⚠️  This test evaluates the existing TF-IDF + Word2Vec hybrid retrieval system")
    print("⚠️  Testing semantic relevance across 50 varied user queries")
    
    input("\nPress ENTER to start the evaluation...")
    
    evaluator = HybridRetrievalSemanticEvaluator()
    
    try:
        results = evaluator.run_semantic_evaluation()
        evaluator.print_results(results)
        
        # Save results to file
        with open('hybrid_retrieval_semantic_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: hybrid_retrieval_semantic_results.json")
        
        # Exit with appropriate code
        sys.exit(0 if results['metrics']['target_achieved'] else 1)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
