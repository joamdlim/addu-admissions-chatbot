#!/usr/bin/env python3
"""
Debug script to investigate ACAT document retrieval issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chroma_db.chroma_service import ChromaService

def main():
    print("🔍 Investigating ACAT Document Retrieval Issue")
    print("=" * 60)
    
    # Get the collection
    collection = ChromaService.get_client().get_or_create_collection(name='addu_admissions_chatbot')
    
    # Get all documents
    all_docs = collection.get(include=['documents', 'metadatas'])
    
    print("\n📋 DOCUMENT TYPES IN DATABASE:")
    doc_types = set()
    for metadata in all_docs.get('metadatas', []):
        if metadata and 'document_type' in metadata:
            doc_types.add(metadata['document_type'])
    
    for doc_type in sorted(doc_types):
        print(f"  - {doc_type}")
    
    print("\n🔍 ACAT-RELATED DOCUMENTS:")
    acat_found = False
    for i, (doc_id, content, metadata) in enumerate(zip(all_docs.get('ids', []), all_docs.get('documents', []), all_docs.get('metadatas', []))):
        if metadata and 'filename' in metadata:
            filename = metadata['filename']
            if 'ACAT' in content.upper() or ('admissions' in filename.lower() and 'aid' in filename.lower()):
                acat_found = True
                print(f"\n📄 Document: {filename}")
                print(f"   Type: {metadata.get('document_type', 'Unknown')}")
                print(f"   Keywords: {metadata.get('keywords', 'None')}")
                print(f"   Content preview: {content[:300]}...")
                print("   " + "-" * 50)
    
    if not acat_found:
        print("❌ No ACAT-related documents found!")
    
    print("\n🎯 ADMISSIONS SPECIALIZED DOCUMENT TYPES:")
    from chatbot.topics import get_retrieval_strategy_config
    strategy_config = get_retrieval_strategy_config('admissions_specialized')
    document_types = strategy_config.get('document_types', [])
    print(f"   Expected types: {document_types}")
    
    print("\n📊 DOCUMENTS BY TYPE:")
    type_counts = {}
    for metadata in all_docs.get('metadatas', []):
        if metadata and 'document_type' in metadata:
            doc_type = metadata['document_type']
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    
    for doc_type, count in sorted(type_counts.items()):
        print(f"   {doc_type}: {count} documents")

if __name__ == "__main__":
    main()
