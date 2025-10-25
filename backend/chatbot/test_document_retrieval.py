#!/usr/bin/env python3
"""
Comprehensive test script for document retrieval improvements.
Tests various query types to ensure the enhanced keyword matching and synonym support
work correctly without breaking existing functionality.
"""

import os
import sys
import django
from typing import List, Dict

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from chatbot.fast_hybrid_chatbot_together import FastHybridChatbotTogether

class DocumentRetrievalTester:
    def __init__(self):
        self.chatbot = FastHybridChatbotTogether(
            use_chroma=True, 
            chroma_collection_name="documents", 
            use_hybrid_topic_retrieval=True
        )
        self.test_results = []
        
    def test_query(self, query: str, expected_doc_type: str = None, description: str = "") -> Dict:
        """Test a single query and return results"""
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {query}")
        print(f"📝 Description: {description}")
        print(f"🎯 Expected doc type: {expected_doc_type}")
        print(f"{'='*60}")
        
        try:
            # Test intent analysis
            intent = self.chatbot.analyze_query_intent(query)
            print(f"🔍 Intent Analysis: {intent}")
            
            # Test document retrieval
            docs = self.chatbot._retrieve_from_chroma(
                query, 
                top_k=3,
                document_type_filter=intent.get('document_type'),
                program_filter=intent.get('program_filter')
            )
            
            print(f"📊 Retrieved {len(docs)} documents:")
            for i, doc in enumerate(docs):
                print(f"   {i+1}. {doc.get('filename', 'N/A')[:50]} | "
                      f"Type: {doc.get('document_type', 'N/A')} | "
                      f"Score: {doc.get('relevance', 0):.3f}")
            
            # Check if expected document type is in results
            doc_types_found = [doc.get('document_type') for doc in docs]
            success = expected_doc_type in doc_types_found if expected_doc_type else len(docs) > 0
            
            result = {
                'query': query,
                'description': description,
                'expected_doc_type': expected_doc_type,
                'intent': intent,
                'docs_found': len(docs),
                'doc_types_found': doc_types_found,
                'success': success,
                'top_doc': docs[0] if docs else None
            }
            
            print(f"✅ SUCCESS" if success else f"❌ FAILED")
            self.test_results.append(result)
            return result
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            result = {
                'query': query,
                'description': description,
                'expected_doc_type': expected_doc_type,
                'error': str(e),
                'success': False
            }
            self.test_results.append(result)
            return result
    
    def run_comprehensive_tests(self):
        """Run comprehensive tests covering all document types and scenarios"""
        print("🚀 Starting Comprehensive Document Retrieval Tests")
        print("=" * 80)
        
        # CONTACT/ADMISSION OFFICE QUERIES (Primary focus)
        print("\n🏢 CONTACT & ADMISSION OFFICE QUERIES")
        self.test_query(
            "Where is the admission office located?",
            "contact",
            "Primary test case - admission office location"
        )
        
        self.test_query(
            "What is the location of the admissions office?",
            "contact", 
            "Variation with 'admissions' plural"
        )
        
        self.test_query(
            "How can I contact the admission office?",
            "contact",
            "Contact information for admission office"
        )
        
        self.test_query(
            "Phone number of admission office",
            "contact",
            "Phone number query"
        )
        
        self.test_query(
            "Email address of admissions office",
            "contact",
            "Email query with admissions plural"
        )
        
        # GENERAL CONTACT QUERIES
        print("\n📞 GENERAL CONTACT QUERIES")
        self.test_query(
            "Contact information",
            "contact",
            "General contact information"
        )
        
        self.test_query(
            "Phone numbers",
            "contact",
            "General phone numbers"
        )
        
        self.test_query(
            "Office locations",
            "contact",
            "General office locations"
        )
        
        # ADMISSION REQUIREMENTS (Should NOT be contact)
        print("\n📋 ADMISSION REQUIREMENTS QUERIES")
        self.test_query(
            "What are the admission requirements?",
            "admission",
            "Should be admission, not contact"
        )
        
        self.test_query(
            "Admission requirements for transfer students",
            "admission",
            "Transfer student requirements"
        )
        
        self.test_query(
            "How to apply for admission",
            "admission",
            "Application process"
        )
        
        # ENROLLMENT QUERIES
        print("\n📝 ENROLLMENT QUERIES")
        self.test_query(
            "Enrollment process",
            "enrollment",
            "General enrollment process"
        )
        
        self.test_query(
            "How to enroll",
            "enrollment",
            "Enrollment how-to"
        )
        
        self.test_query(
            "Registration procedures",
            "enrollment",
            "Registration procedures"
        )
        
        # SCHOLARSHIP QUERIES
        print("\n💰 SCHOLARSHIP QUERIES")
        self.test_query(
            "Available scholarships",
            "scholarship",
            "Scholarship information"
        )
        
        self.test_query(
            "Financial aid options",
            "scholarship",
            "Financial aid"
        )
        
        self.test_query(
            "Scholarship requirements",
            "scholarship",
            "Scholarship requirements"
        )
        
        # ACADEMIC PROGRAM QUERIES
        print("\n🎓 ACADEMIC PROGRAM QUERIES")
        self.test_query(
            "Computer Science program",
            "academic",
            "Specific program query"
        )
        
        self.test_query(
            "Available degree programs",
            "academic",
            "General programs query"
        )
        
        self.test_query(
            "Course curriculum",
            "academic",
            "Curriculum information"
        )
        
        # FEES QUERIES
        print("\n💵 FEES QUERIES")
        self.test_query(
            "Tuition fees",
            "fees",
            "Tuition information"
        )
        
        self.test_query(
            "Payment options",
            "fees",
            "Payment information"
        )
        
        self.test_query(
            "Cost of education",
            "fees",
            "Cost information"
        )
        
        # EDGE CASES
        print("\n🔍 EDGE CASES")
        self.test_query(
            "Where can I find the registrar office?",
            "contact",
            "Registrar office location (should be contact)"
        )
        
        self.test_query(
            "Registrar requirements",
            None,  # Could be admission or other
            "Registrar requirements (ambiguous)"
        )
        
        self.test_query(
            "Office hours",
            "contact",
            "Office hours query"
        )
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result.get('success', False))
        failed_tests = total_tests - successful_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Successful: {successful_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result.get('success', False):
                    print(f"   - {result['query']}")
                    if 'error' in result:
                        print(f"     Error: {result['error']}")
                    else:
                        print(f"     Expected: {result['expected_doc_type']}, "
                              f"Found: {result.get('doc_types_found', [])}")
        
        print(f"\n🎯 CONTACT QUERY TESTS:")
        contact_tests = [r for r in self.test_results if r.get('expected_doc_type') == 'contact']
        contact_success = sum(1 for r in contact_tests if r.get('success', False))
        print(f"Contact Tests: {len(contact_tests)}")
        print(f"Contact Success: {contact_success}/{len(contact_tests)}")
        
        if contact_success < len(contact_tests):
            print("⚠️  Some contact queries failed - this indicates the main issue may not be fully resolved")
        else:
            print("✅ All contact queries successful - admission office location issue should be fixed!")

def main():
    """Main test runner"""
    tester = DocumentRetrievalTester()
    
    try:
        tester.run_comprehensive_tests()
        tester.print_summary()
        
        # Return appropriate exit code
        failed_tests = sum(1 for result in tester.test_results if not result.get('success', False))
        sys.exit(0 if failed_tests == 0 else 1)
        
    except Exception as e:
        print(f"❌ Test runner failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
