#!/usr/bin/env python3
import sys
sys.path.append('.')
from backend.chatbot.chroma_connection import ChromaService

def main():
    print("[TEST] Testing ACAT document retrieval")
    print("=" * 50)
    
    try:
        # Get the documents collection
        client = ChromaService.get_client()
        collection = client.get_collection(name='documents')
        
        print(f"[OK] Using collection: {collection.name}")
        print(f"[INFO] Total documents: {collection.count()}")
        
        # Search for ACAT
        print(f"\n[SEARCH] Searching for ACAT...")
        
        # Get all documents and search for ACAT
        all_docs = collection.get(include=['documents', 'metadatas'])
        
        acat_docs = []
        for i, (doc_id, content, metadata) in enumerate(zip(all_docs.get('ids', []), all_docs.get('documents', []), all_docs.get('metadatas', []))):
            if content and 'ACAT' in content.upper():
                acat_docs.append({
                    'id': doc_id,
                    'filename': metadata.get('filename', 'Unknown') if metadata else 'Unknown',
                    'document_type': metadata.get('document_type', 'Unknown') if metadata else 'Unknown',
                    'keywords': metadata.get('keywords', 'None') if metadata else 'None',
                    'content_preview': content[:200] + '...' if len(content) > 200 else content
                })
        
        if acat_docs:
            print(f"[SUCCESS] Found {len(acat_docs)} ACAT-related documents:")
            for i, doc in enumerate(acat_docs):
                print(f"\n  {i+1}. {doc['filename']}")
                print(f"     Type: {doc['document_type']}")
                print(f"     Keywords: {doc['keywords']}")
                print(f"     Content: {doc['content_preview']}")
        else:
            print("[ERROR] No ACAT documents found")
            
        # Check if the document type is included in admissions strategy
        print(f"\n[CHECK] Verifying document type inclusion...")
        from backend.chatbot.topics import get_retrieval_strategy_config
        strategy_config = get_retrieval_strategy_config('admissions_specialized')
        document_types = strategy_config.get('document_types', [])
        print(f"Admissions strategy includes: {document_types}")
        
        for doc in acat_docs:
            if doc['document_type'] in document_types:
                print(f"[OK] {doc['filename']} type '{doc['document_type']}' is included in admissions strategy")
            else:
                print(f"[WARNING] {doc['filename']} type '{doc['document_type']}' is NOT included in admissions strategy")
                
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
