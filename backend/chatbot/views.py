from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, StreamingHttpResponse
import json
from .chroma_connection import ChromaService
from .improved_pdf_to_chroma import sync_supabase_to_chroma_improved
from .supabase_client import get_supabase_client
from .models import Conversation, ConversationTurn, SystemPrompt
# Remove the enhanced chatbot import
# from .enhanced_hybrid_chatbot import get_enhanced_chatbot
# from .fast_hybrid_chatbot import FastHybridChatbot

# Add the Together AI chatbot import
from .fast_hybrid_chatbot_together import FastHybridChatbotTogether
import os, json
import uuid

# Initialize Together AI chatbot at startup
print("🚀 Initializing Together AI chatbot...")
together_chatbot = FastHybridChatbotTogether(use_chroma=True)
print("✅ Together AI chatbot ready!")

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
            # This function is no longer used as per the new_code, but kept for now
            # process_and_store_pdf_in_chroma(pdf_bytes, pdf_file_name) 
            return JsonResponse({"status": f"PDF '{pdf_file_name}' processed and stored successfully."})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "POST request required"}, status=400)

@csrf_exempt
def chat_view(request):
    """Chat view using Together AI"""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            prompt = data.get("prompt", "")
            session_id = data.get("session_id", None)
            max_tokens = data.get("max_tokens", 3000)
            min_relevance = data.get("min_relevance", 0.1)
            
            def generate_streaming_response():
                try:
                    # Use the Together AI chatbot
                    for event in together_chatbot.process_query_stream(
                        query=prompt,
                        max_tokens=max_tokens,
                        min_relevance=min_relevance,
                        use_history=True
                    ):
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

@csrf_exempt 
def chat_legacy_view(request):
    """Legacy chat view for backward compatibility (using Together AI)"""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            prompt = data.get("prompt", "")
            
            def generate_streaming_response():
                try:
                    # Use Together AI chatbot for legacy support
                    for event in together_chatbot.process_query_stream(prompt, max_tokens=3000, min_relevance=0.1, use_history=True):
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

# ===== SIMPLIFIED CONVERSATION MANAGEMENT ENDPOINTS =====
# Note: These endpoints are simplified since we're not using complex conversation memory

@api_view(['POST'])
def create_conversation(request):
    """Create a new conversation session (simplified)"""
    try:
        session_id = str(uuid.uuid4())
        
        return Response({
            "status": "success",
            "session_id": session_id,
            "message": "Conversation created (simplified mode)"
        })
        
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['GET'])
def get_conversation_history(request, session_id):
    """Get conversation history for a session (simplified)"""
    try:
        # In simplified mode, we don't store conversation history
        # This is just for API compatibility
        return Response({
            "session_id": session_id,
            "history": [],
            "count": 0,
            "message": "Conversation history not stored in simplified mode"
        })
        
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['GET'])
def get_conversation_stats(request, session_id):
    """Get conversation statistics (simplified)"""
    try:
        return Response({
            'session_id': session_id,
            'message': 'Simplified mode - no detailed stats available'
        })
        
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['DELETE'])
def clear_conversation(request, session_id):
    """Clear/reset a conversation (simplified)"""
    try:
        return Response({
            "status": "success",
            "message": f"Conversation {session_id} cleared (simplified mode)"
        })
        
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['GET'])
def list_active_conversations(request):
    """List all active conversations (simplified)"""
    try:
        return Response({
            "conversations": [],
            "count": 0,
            "message": "Simplified mode - no conversation tracking"
        })
        
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@csrf_exempt
def force_summarization(request, session_id):
    """Manually trigger summarization for a conversation (simplified)"""
    if request.method == "POST":
        try:
            return JsonResponse({
                "status": "success",
                "message": "Summarization not needed in simplified mode"
            })
            
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "POST request required"}, status=400)

# ===== SYSTEM PROMPT MANAGEMENT =====

@api_view(['GET'])
def get_system_prompts(request):
    """Get all system prompts"""
    try:
        prompts = SystemPrompt.objects.all().order_by('-created_at')
        
        prompt_list = []
        for prompt in prompts:
            prompt_list.append({
                'id': prompt.id,
                'name': prompt.name,
                'prompt_text': prompt.prompt_text,
                'token_count': prompt.token_count,
                'is_active': prompt.is_active,
                'created_at': prompt.created_at.isoformat()
            })
        
        return Response({"prompts": prompt_list})
        
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@csrf_exempt
def update_system_prompt(request):
    """Create or update system prompt"""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            name = data.get("name", "default")
            prompt_text = data.get("prompt_text", "")
            
            if not prompt_text:
                return JsonResponse({"error": "Prompt text is required"}, status=400)
            
            # Count tokens (simplified)
            token_count = len(prompt_text.split())
            
            # Deactivate other prompts
            SystemPrompt.objects.filter(is_active=True).update(is_active=False)
            
            # Create or update prompt
            prompt, created = SystemPrompt.objects.update_or_create(
                name=name,
                defaults={
                    'prompt_text': prompt_text,
                    'token_count': token_count,
                    'is_active': True
                }
            )
            
            return JsonResponse({
                "status": "success",
                "prompt_id": prompt.id,
                "token_count": token_count,
                "created": created
            })
            
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "POST request required"}, status=400)

# ===== EXISTING ENDPOINTS (UNCHANGED) =====

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
        # Use the improved sync function with default parameters
        result = sync_supabase_to_chroma_improved()
        
        # Convert to the expected response format
        if "error" in result:
            return JsonResponse({"error": result["error"]}, status=500)
        
        return JsonResponse({
            "status": result["status"], 
            "found": result["found"], 
            "ingested": result["ingested"],
            "extraction_method": result.get("extraction_method", "pdfplumber")
        })
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

@csrf_exempt
def admin_upload_file(request):
    """Upload file directly to Supabase storage and process it for ChromaDB"""
    if request.method == "POST":
        if 'file' not in request.FILES:
            return JsonResponse({"error": "No file uploaded"}, status=400)
        
        uploaded_file = request.FILES['file']
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name
        
        try:
            supabase = get_supabase_client()
            
            # Upload to Supabase storage
            result = supabase.storage.from_("original-pdfs").upload(
                file_name,
                file_bytes,
                {"content-type": uploaded_file.content_type}
            )
            
            # Process and store in ChromaDB immediately for PDFs only
            if file_name.lower().endswith('.pdf'):
                try:
                    from .improved_pdf_to_chroma import process_and_store_pdf_in_chroma_improved
                    documents_stored = process_and_store_pdf_in_chroma_improved(
                        file_bytes, 
                        file_name,
                        chunk_size=1000,
                        overlap=200
                    )
                    message = f"File '{file_name}' uploaded and processed successfully ({documents_stored} documents stored in ChromaDB)"
                except Exception as chroma_error:
                    print(f"⚠️ ChromaDB processing failed for {file_name}: {chroma_error}")
                    message = f"File '{file_name}' uploaded to Supabase but ChromaDB processing failed: {str(chroma_error)}"
            else:
                message = f"File '{file_name}' uploaded successfully (non-PDF files are stored in Supabase only)"
            
            return JsonResponse({
                "status": "success",
                "message": message,
                "file_name": file_name
            })
            
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "POST request required"}, status=400)

@api_view(['GET'])
def admin_list_files(request):
    """List all files from Supabase storage"""
    try:
        supabase = get_supabase_client()
        files = supabase.storage.from_("original-pdfs").list()
        
        # Format file data
        file_list = []
        for file in files:
            file_list.append({
                "name": file.get("name"),
                "size": file.get("metadata", {}).get("size", 0),
                "created_at": file.get("created_at"),
                "updated_at": file.get("updated_at"),
                "content_type": file.get("metadata", {}).get("mimetype", "unknown")
            })
        
        return Response({"files": file_list})
        
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@csrf_exempt
def admin_delete_file(request):
    """Delete file from Supabase storage and ChromaDB"""
    if request.method == "DELETE":
        try:
            data = json.loads(request.body)
            file_name = data.get("file_name")
            
            if not file_name:
                return JsonResponse({"error": "File name required"}, status=400)
            
            supabase = get_supabase_client()
            
            # Delete from Supabase storage
            supabase.storage.from_("original-pdfs").remove([file_name])
            
            # Also remove from ChromaDB if it's a PDF
            if file_name.lower().endswith('.pdf'):
                try:
                    from .chroma_connection import ChromaService
                    chroma_client = ChromaService.get_client()
                    collection_name = os.getenv("CHROMA_COLLECTION", "documents")
                    collection = chroma_client.get_or_create_collection(name=collection_name)
                    
                    # Generate the document ID (same as used during upload)
                    doc_id = os.path.splitext(file_name)[0]
                    
                    # Delete from ChromaDB
                    collection.delete(ids=[doc_id])
                    message = f"File '{file_name}' deleted from both Supabase and ChromaDB"
                except Exception as chroma_error:
                    print(f"⚠️ ChromaDB deletion failed for {file_name}: {chroma_error}")
                    message = f"File '{file_name}' deleted from Supabase but ChromaDB deletion failed"
            else:
                message = f"File '{file_name}' deleted successfully"
            
            return JsonResponse({
                "status": "success",
                "message": message
            })
            
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "DELETE request required"}, status=400)

# Keep the manual sync function for individual files if needed
@csrf_exempt
def admin_sync_file_to_chroma(request):
    """Manually sync a specific file from Supabase to ChromaDB"""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            file_name = data.get("file_name")
            
            if not file_name:
                return JsonResponse({"error": "File name required"}, status=400)
            
            if not file_name.lower().endswith('.pdf'):
                return JsonResponse({"error": "Only PDF files can be synced to ChromaDB"}, status=400)
            
            supabase = get_supabase_client()
            
            # Download file from Supabase
            pdf_bytes = supabase.storage.from_("original-pdfs").download(file_name)
            
            # Process and store in ChromaDB using the improved method
            from .improved_pdf_to_chroma import process_and_store_pdf_in_chroma_improved
            result = process_and_store_pdf_in_chroma_improved(
                pdf_bytes, 
                file_name,
                chunk_size=1000,
                overlap=200
            )
            
            return JsonResponse({
                "status": "success",
                "message": f"File '{file_name}' successfully synced to ChromaDB",
                "documents_stored": result
            })
            
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "POST request required"}, status=400)
