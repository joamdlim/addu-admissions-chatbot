import os
import json

# Create processed directory if it doesn't exist
os.makedirs("processed", exist_ok=True)

# Create sample documents
sample_docs = [
    {
        "title": "Sample Document 1",
        "content": "This is a sample document about machine learning and artificial intelligence.",
        "source": "example.com"
    },
    {
        "title": "Sample Document 2",
        "content": "Natural language processing is a subfield of artificial intelligence.",
        "source": "example.org"
    },
    {
        "title": "Sample Document 3",
        "content": "Vector embeddings are used to represent text as numerical vectors.",
        "source": "example.net"
    }
]

# Save each document as a separate JSON file
for i, doc in enumerate(sample_docs):
    with open(f"processed/doc_{i+1}.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

print(f"✅ Created {len(sample_docs)} sample documents in the 'processed' directory")