import os
import numpy as np
import json
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# Change these relative imports to absolute imports
from vectorizer import vectorize_documents, compute_word2vec_vector, build_tfidf_vectorizer
from llama_interface import llm, correct_typos
from preprocess import preprocess_text

class HybridChatbot:
    def __init__(self, embeddings_dir="../embeddings", processed_dir="../processed"):
        self.vectors_path = os.path.join(embeddings_dir, "hybrid_vectors.npy")
        self.metadata_path = os.path.join(embeddings_dir, "metadata.json")
        self.processed_dir = processed_dir
        
        # Load vectors and metadata
        print("🔄 Loading vectors and metadata...")
        self.vectors = None
        self.documents = []
        self._load_vectors_and_metadata()
        
        # Initialize TFIDF vectorizer from documents
        if self.documents:
            corpus = [" ".join(preprocess_text(doc["content"])) for doc in self.documents]
            self.tfidf_vectorizer = build_tfidf_vectorizer(corpus)
            print(f"✅ TFIDF Vectorizer built on {len(corpus)} documents")
        else:
            print("⚠️ No documents to build TFIDF vectorizer")
    
    def _load_vectors_and_metadata(self):
        """Load the vectors and metadata from disk"""
        try:
            if os.path.exists(self.vectors_path):
                self.vectors = np.load(self.vectors_path)
                print(f"✅ Loaded vectors with shape: {self.vectors.shape}")
            else:
                print("⚠️ Vectors file not found")
                
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
                print(f"✅ Loaded metadata for {len(self.documents)} documents")
            else:
                print("⚠️ Metadata file not found")
                
            # Verify alignment
            if self.vectors is not None and len(self.documents) != self.vectors.shape[0]:
                print(f"⚠️ Vectors count ({self.vectors.shape[0]}) doesn't match document count ({len(self.documents)})")
                
        except Exception as e:
            print(f"❌ Error loading vectors/metadata: {e}")
    
    def _vectorize_query(self, query: str) -> np.ndarray:
        """Convert user query to the same hybrid vector space as documents"""
        processed_query = preprocess_text(query)
        query_text = " ".join(processed_query)
        
        # Get TFIDF vector
        tfidf_vector = self.tfidf_vectorizer.transform([query_text]).toarray()[0]
        
        # Get Word2Vec vector
        w2v_vector = compute_word2vec_vector(processed_query)
        
        # Combine them
        hybrid_vector = np.concatenate((tfidf_vector, w2v_vector))
        return hybrid_vector
    
    def _retrieve_relevant_docs(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict]:
        """Find most relevant documents using cosine similarity"""
        if self.vectors is None or len(self.documents) == 0:
            return []
            
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector.reshape(1, -1), self.vectors)[0]
        
        # Get top K matches
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        relevant_docs = []
        for idx in top_indices:
            similarity_score = similarities[idx]
            if similarity_score > 0.1:  # Only include if relevance is above threshold
                doc = self.documents[idx].copy()
                doc["relevance"] = float(similarity_score)
                relevant_docs.append(doc)
                
        return relevant_docs
    
    def _generate_response(self, query: str, relevant_docs: List[Dict]) -> str:
        """Generate a response using Llama and retrieved documents"""
        # Format retrieved context
        context = "\n\n".join([f"Document: {doc['content']}" for doc in relevant_docs])
        
        # Create a prompt for Llama
        prompt = f"""
        You are a helpful assistant for Ateneo de Davao University.
        
        Here is information that might be relevant to the user's query:
        {context}
        
        User question: {query}
        
        Answer the user's question based on the information provided above. If the information doesn't contain the answer, say you don't know but try to be helpful. Be concise and specific in your response.
        
        Answer:
        """
        
        # Generate response using the Llama model
        try:
            result = llm(prompt, max_tokens=150, echo=False, temperature=0.7)
            response = result["choices"][0]["text"].strip()
            return response
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "I'm sorry, I had trouble generating a response. Please try again."
    
    def process_query(self, query: str) -> Tuple[str, List[Dict]]:
        """Main method to process a user query and return a response"""
        # First, correct any typos in the query
        corrected_query = correct_typos(query)
        
        # If query was corrected, note that
        query_info = ""
        if corrected_query.lower() != query.lower():
            query_info = f"Corrected query: {corrected_query}\n"
            query = corrected_query
        
        # Vectorize the query
        query_vector = self._vectorize_query(query)
        
        # Retrieve relevant documents
        relevant_docs = self._retrieve_relevant_docs(query_vector)
        
        # Generate response
        response = self._generate_response(query, relevant_docs)
        
        return response, relevant_docs

# Create a test function to demo the hybrid chatbot
def test_hybrid_chatbot():
    chatbot = HybridChatbot()
    
    test_queries = [
        "How do I apply for admission at Ateneo de Davao?",
        "What documents do I need for enrollment?",
        "When is the entrance exam?"
    ]
    
    print("\n🤖 HYBRID CHATBOT TEST 🤖\n")
    
    for query in test_queries:
        print(f"🔍 Query: {query}")
        
        response, relevant_docs = chatbot.process_query(query)
        
        print(f"📚 Retrieved {len(relevant_docs)} relevant document(s)")
        for i, doc in enumerate(relevant_docs):
            print(f"  [{i+1}] {doc['id']} (Relevance: {doc['relevance']:.2f})")
        
        print(f"💬 Response: {response}\n")
        print("-" * 80)

if __name__ == "__main__":
    test_hybrid_chatbot()