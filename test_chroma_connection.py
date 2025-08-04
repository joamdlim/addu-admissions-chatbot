import chromadb
from chromadb.config import Settings

def test_chroma_connection():
    try:
        # Initialize the ChromaDB client
        chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"  # Optional: Directory for persistent storage
        ))

        # Create a test collection
        collection = chroma_client.create_collection(name="test_collection")

        # Add a test document
        collection.add(
            ids=["test_id"],
            documents=["This is a test document."],
            metadatas=[{"source": "test"}]
        )

        # Query the collection
        results = collection.query(
            query_texts=["test document"],
            n_results=1
        )

        print("ChromaDB connection successful!")
        print("Query results:", results)
        return True
    except Exception as e:
        print("ChromaDB connection failed:", e)
        return False

if __name__ == "__main__":
    test_chroma_connection() 