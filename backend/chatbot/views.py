from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from chatbot.fast_hybrid_chatbot import FastHybridChatbot
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, StreamingHttpResponse
import json
from .chroma_connection import ChromaService
from .test_pdf_to_chroma import process_and_store_pdf_in_chroma
from .supabase_client import get_supabase_client
import os, json

# Initialize the chatbot
chatbot = FastHybridChatbot(use_chroma=True)

@api_view(['GET'])
def evaluate(request):
    return Response({"f1": 0.85, "bleu": 0.72, "satisfaction": 4.2})

@csrf_exempt
def upload_pdf_view(request):
    if request.method == "POST":
        if 'pdf_file' not in request.FILES:
            return JsonResponse({"error": "No file uploaded"}, status=400)
        
        pdf_file = request.FILES['pdf_file']
        pdf_bytes = pdf_file.read()
        pdf_file_name = pdf_file.name

        try:
            process_and_store_pdf_in_chroma(pdf_bytes, pdf_file_name)
            return JsonResponse({"status": f"PDF '{pdf_file_name}' processed and stored successfully."})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "POST request required"}, status=400)

@csrf_exempt
def chat_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            prompt = data.get("prompt", "")
            
            def generate_streaming_response():
                try:
                    # Use the GLOBAL chatbot instance (no reloading!)
                    for event in chatbot.process_query_stream(prompt, min_relevance=0.3, use_history=True):  # Increase from 0.1 to 0.3
                        yield "data: " + json.dumps(event) + "\n\n"
                        
                except Exception as e:
                    yield "data: " + json.dumps({"error": str(e)}) + "\n\n"
                    yield "data: " + json.dumps({"done": True}) + "\n\n"
            
            response = StreamingHttpResponse(
                generate_streaming_response(),
                content_type='text/event-stream'
            )
            
            response['Cache-Control'] = 'no-cache'
            response['Access-Control-Allow-Origin'] = '*'
            response['Access-Control-Allow-Headers'] = 'Content-Type'
            
            return response
            
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "POST request required"}, status=400)


def chroma_test_add(request):
    collection = ChromaService.get_collection()
    
    # Dummy embedding
    collection.add(
        ids=["test-id"],
        documents=["Hello Chroma from Django"],
        embeddings=[[0.1, 0.2, 0.3, 0.4]]
    )
    
    return JsonResponse({"status": "Document added"})

def chroma_test_query(request):
    collection = ChromaService.get_collection()
    
    results = collection.query(
        query_embeddings=[[0.1, 0.2, 0.3, 0.4]],
        n_results=1
    )
    
    return JsonResponse({"results": results})

@csrf_exempt
def sync_supabase_to_chroma(request):
    try:
        supabase = get_supabase_client()
        files = supabase.storage.from_("original-pdfs").list()
        pdf_names = [f['name'] for f in files if f['name'].lower().endswith('.pdf')]

        ingested = 0
        for pdf_file in pdf_names:
            pdf_bytes = supabase.storage.from_("original-pdfs").download(pdf_file)
            try:
                process_and_store_pdf_in_chroma(pdf_bytes, pdf_file)
                ingested += 1
            except Exception as e:
                print(f"Failed to ingest {pdf_file}: {e}")
                continue

        return JsonResponse({"status": "ok", "found": len(pdf_names), "ingested": ingested})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@api_view(['GET'])
def export_chroma_to_local(request):
    try:
        client = ChromaService.get_client()
        collection_name = os.getenv("CHROMA_COLLECTION", "documents")
        col = client.get_or_create_collection(name=collection_name)

        # Pull all items (paginate)
        all_docs = []
        limit = 1000
        offset = 0
        while True:
            res = col.get(include=["documents", "metadatas"], limit=limit, offset=offset)
            ids = res.get("ids", [])
            docs = res.get("documents", [])
            if not ids:
                break
            for i, doc_id in enumerate(ids):
                content = docs[i] if i < len(docs) else ""
                all_docs.append({"id": doc_id, "content": content})
            offset += len(ids)

        # Paths relative to backend/
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        processed_dir = os.path.join(base_dir, "processed")
        embeddings_dir = os.path.join(base_dir, "embeddings")
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(embeddings_dir, exist_ok=True)

        processed_path = os.path.join(processed_dir, "processed.json")
        metadata_path = os.path.join(embeddings_dir, "metadata.json")

        with open(processed_path, "w", encoding="utf-8") as f:
            json.dump(all_docs, f, ensure_ascii=False, indent=2)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(all_docs, f, ensure_ascii=False, indent=2)

        # Optional: force local vector regeneration next time by removing vectors file
        vectors_path = os.path.join(embeddings_dir, "hybrid_vectors.npy")
        if os.path.exists(vectors_path):
            os.remove(vectors_path)

        return JsonResponse({"status": "ok", "exported": len(all_docs), "collection": collection_name})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
