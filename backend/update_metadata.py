# backend/update_metadata.py
import json
import os
import numpy as np
import shutil

# Define paths
processed_path = "processed/processed.json"
metadata_path = "embeddings/metadata.json"
vectors_path = "embeddings/hybrid_vectors.npy"

# Backup existing files
if os.path.exists(metadata_path):
    shutil.copy(metadata_path, f"{metadata_path}.bak")
    print(f"✅ Backed up {metadata_path} to {metadata_path}.bak")

if os.path.exists(vectors_path):
    shutil.copy(vectors_path, f"{vectors_path}.bak")
    print(f"✅ Backed up {vectors_path} to {vectors_path}.bak")

# Load processed documents
with open(processed_path, 'r', encoding='utf-8') as f:
    documents = json.load(f)
    print(f"✅ Loaded {len(documents)} documents from {processed_path}")

# Update metadata.json
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(documents, f, indent=2)
    print(f"✅ Updated {metadata_path} with {len(documents)} documents")

# Delete vectors file to force regeneration
if os.path.exists(vectors_path):
    os.remove(vectors_path)
    print(f"✅ Removed {vectors_path} (will be regenerated on next run)")

print("\n✅ Done! Run your hybrid_chatbot.py script to regenerate vectors and test.")