#!/usr/bin/env python3
"""
Quick test script specifically for contact/admission office queries.
Tests the improvements made to handle "where is the admission office" type queries.
"""

import os
import sys
import django

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from chatbot.fast_hybrid_chatbot_together import FastHybridChatbotTogether

def test_contact_queries():
    """Test contact and admission office queries"""
    print("🧪 Testing Contact & Admission Office Queries")
    print("=" * 60)
    
    chatbot = FastHybridChatbotTogether(
        use_chroma=True, 
        chroma_collection_name="documents", 
        use_hybrid_topic_retrieval=True
    )
    
    # Test queries that should retrieve contact information
    test_queries = [
        "Where is the admission office located?",
        "What is the location of the admissions office?", 
        "How can I contact the admission office?",
        "Phone number of admission office",
        "Email address of admissions office",
        "Contact information for admission office",
        "Where can I find the registrar office?",
        "Office locations",
        "Contact information"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing: '{query}'")
        print("-" * 50)
        
        # Test intent analysis
        intent = chatbot.analyze_query_intent(query)
        print(f"🎯 Intent: {intent}")
        
        # Test document retrieval
        try:
            docs = chatbot._retrieve_from_chroma(
                query, 
                top_k=3,
                document_type_filter=intent.get('document_type')
            )
            
            print(f"📊 Found {len(docs)} documents:")
            for j, doc in enumerate(docs):
                print(f"   {j+1}. {doc.get('filename', 'N/A')[:60]}")
                print(f"      Type: {doc.get('document_type', 'N/A')} | Score: {doc.get('relevance', 0):.3f}")
                print(f"      Keywords: {doc.get('keywords', 'N/A')[:80]}")
            
            # Check if we got contact documents
            contact_docs = [doc for doc in docs if doc.get('document_type') == 'contact']
            if contact_docs:
                print(f"✅ SUCCESS: Found {len(contact_docs)} contact document(s)")
            else:
                print(f"❌ FAILED: No contact documents found")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\n{'='*60}")
    print("🏁 Contact Query Tests Complete")
    print("If you see contact documents being retrieved for admission office queries,")
    print("then the issue should be resolved!")

def test_synonym_expansion():
    """Test the synonym expansion functionality"""
    print("\n🔍 Testing Synonym Expansion")
    print("=" * 40)
    
    chatbot = FastHybridChatbotTogether(
        use_chroma=True, 
        chroma_collection_name="documents", 
        use_hybrid_topic_retrieval=True
    )
    
    # Test synonym expansion
    test_terms = [
        {'admission', 'office', 'location'},
        {'contact', 'phone'},
        {'registrar', 'office'},
        {'enrollment', 'process'}
    ]
    
    for terms in test_terms:
        expanded = chatbot._expand_query_terms_with_synonyms(terms)
        print(f"Original: {terms}")
        print(f"Expanded: {expanded}")
        print(f"Added: {expanded - terms}")
        print()

if __name__ == "__main__":
    test_contact_queries()
    test_synonym_expansion()
