from chatbot.vectorizer import load_json_docs, vectorize_documents
import numpy as np
import json
import os

# Step 1: Load JSON documents from processed/
folder_path = os.path.join("processed")
documents = load_json_docs(folder_path)

# Step 2: Generate hybrid vectors
vectors, metadata = vectorize_documents(documents)

# Step 3: Print summary
print(f"✅ Loaded {len(vectors)} document(s)")
print(f"🔢 Vector dimension: {vectors[0].shape}")
print(f"🧾 Sample metadata: {metadata[0]}")

# Step 4: Save output (optional)
os.makedirs("embeddings", exist_ok=True)
np.save("embeddings/hybrid_vectors.npy", vectors)

with open("embeddings/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
