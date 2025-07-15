"""
Ultra-optimized version of the hybrid chatbot for maximum speed.
This version prioritizes speed over quality.
"""

import os
import sys
import numpy as np
import json
import time
from typing import List, Dict, Tuple

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the optimized interface
try:
    from chatbot.llama_interface_optimized import llm, generate_fast_response
except ImportError:
    from chatbot.llama_interface_optimized import llm, generate_fast_response

class FastChatbot:
    """Ultra-optimized chatbot for maximum speed"""
    
    def __init__(self, embeddings_dir="../embeddings", processed_dir="../processed"):
        self.metadata_path = os.path.join(embeddings_dir, "metadata.json")
        self.documents = []
        self._load_metadata()
        
        print("⚡ Fast chatbot initialized")
    
    def _load_metadata(self):
        """Load only the metadata (skip vectors for speed)"""
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                print(f"✅ Loaded metadata for {len(self.documents)} documents")
            else:
                print("⚠️ Metadata file not found")
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
    
    def search_documents(self, query: str, max_docs: int = 2) -> List[Dict]:
        """Simple keyword search for speed"""
        if not self.documents:
            return []
        
        # Convert query to lowercase for case-insensitive matching
        query_terms = query.lower().split()
        
        # Score documents based on keyword matches
        scored_docs = []
        for doc in self.documents:
            content = doc['content'].lower()
            score = sum(content.count(term) for term in query_terms)
            
            if score > 0:
                scored_docs.append({
                    'id': doc['id'],
                    'content': doc['content'],
                    'relevance': score
                })
        
        # Sort by relevance and take top results
        return sorted(scored_docs, key=lambda x: x['relevance'], reverse=True)[:max_docs]
    
    def process_query(self, query: str) -> Tuple[str, List[Dict]]:
        """Process a query with maximum speed"""
        # Search for relevant documents
        relevant_docs = self.search_documents(query)
        
        # Format context - ultra minimal
        context = " ".join([doc['content'][:100] for doc in relevant_docs])
        
        # Create minimal prompt
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        
        # Generate response with optimized settings
        response = generate_fast_response(prompt, max_tokens=500)
        
        return response, relevant_docs

def test_optimized_chatbot():
    """Test the optimized chatbot"""
    chatbot = FastChatbot()
    
    print("\n⚡⚡⚡ ULTRA-FAST CHATBOT MODE ⚡⚡⚡")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("🔍 Query: ")
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("👋 Goodbye!")
            break
        
        # Time the entire process
        start_time = time.time()
        
        # Process query
        response, relevant_docs = chatbot.process_query(query)
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        
        print(f"\n⏱️ Total response time: {elapsed:.2f} seconds")
        print(f"📚 Found {len(relevant_docs)} relevant documents")
        
        # Show response
        print("\n💬 Response: ", end="")
        print(response)
        
        print("\n")
        print("-" * 80)

if __name__ == "__main__":
    test_optimized_chatbot() 