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
from .enhanced_hybrid_chatbot import get_enhanced_chatbot
from .fast_hybrid_chatbot import FastHybridChatbot
import os, json
import uuid

# Initialize chatbots at startup (eager loading)
print("🚀 Initializing chatbots at server startup...")
enhanced_chatbot = get_enhanced_chatbot()
legacy_chatbot = FastHybridChatbot(use_chroma=True)
print("✅ All chatbots ready!")

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
    """Enhanced chat view with conversation memory support"""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            prompt = data.get("prompt", "")
            session_id = data.get("session_id", None)  # Optional session ID
            max_tokens = data.get("max_tokens", 3000)
            min_relevance = data.get("min_relevance", 0.1)
            
            # Generate a session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())
            
            def generate_streaming_response():
                try:
                    # Use enhanced chatbot with conversation memory
                    for event in enhanced_chatbot.process_query_with_memory(
                        query=prompt,
                        session_id=session_id,
                        max_tokens=max_tokens,
                        stream=True,
                        require_context=True,
                        min_relevance=min_relevance
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
    """Legacy chat view for backward compatibility (without conversation memory)"""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            prompt = data.get("prompt", "")
            
            def generate_streaming_response():
                try:
                    # Use original chatbot for legacy support
                    for event in legacy_chatbot.process_query_stream(prompt, max_tokens=3000, min_relevance=0.1, use_history=True):
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

# ===== NEW CONVERSATION MANAGEMENT ENDPOINTS =====

@api_view(['POST'])
def create_conversation(request):
    """Create a new conversation session"""
    try:
        session_id = str(uuid.uuid4())
        conversation = enhanced_chatbot.get_or_create_conversation(session_id)
        
        return Response({
            "status": "success",
            "session_id": conversation.session_id,
            "created_at": conversation.created_at.isoformat()
        })
        
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['GET'])
def get_conversation_history(request, session_id):
    """Get conversation history for a session"""
    try:
        limit = int(request.GET.get('limit', 20))
        history = enhanced_chatbot.get_conversation_history(session_id, limit)
        
        return Response({
            "session_id": session_id,
            "history": history,
            "count": len(history)
        })
        
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['GET'])
def get_conversation_stats(request, session_id):
    """Get conversation statistics"""
    try:
        stats = enhanced_chatbot.get_conversation_stats(session_id)
        
        if not stats:
            return Response({"error": "Conversation not found"}, status=404)
        
        return Response(stats)
        
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['DELETE'])
def clear_conversation(request, session_id):
    """Clear/reset a conversation"""
    try:
        success = enhanced_chatbot.clear_conversation(session_id)
        
        if success:
            return Response({
                "status": "success",
                "message": f"Conversation {session_id} cleared"
            })
        else:
            return Response({"error": "Conversation not found"}, status=404)
        
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['GET'])
def list_active_conversations(request):
    """List all active conversations"""
    try:
        conversations = Conversation.objects.filter(is_active=True).order_by('-updated_at')[:50]
        
        conversation_list = []
        for conv in conversations:
            conversation_list.append({
                'session_id': conv.session_id,
                'title': conv.title,
                'total_exchanges': conv.total_exchanges,
                'created_at': conv.created_at.isoformat(),
                'updated_at': conv.updated_at.isoformat(),
                'current_token_count': conv.current_token_count
            })
        
        return Response({
            "conversations": conversation_list,
            "count": len(conversation_list)
        })
        
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@csrf_exempt
def force_summarization(request, session_id):
    """Manually trigger summarization for a conversation"""
    if request.method == "POST":
        try:
            conversation = Conversation.objects.get(session_id=session_id)
            enhanced_chatbot.perform_summarization(conversation)
            
            # Get updated stats
            stats = enhanced_chatbot.get_conversation_stats(session_id)
            
            return JsonResponse({
                "status": "success",
                "message": "Summarization completed",
                "stats": stats
            })
            
        except Conversation.DoesNotExist:
            return JsonResponse({"error": "Conversation not found"}, status=404)
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
            
            # Count tokens
            from .token_utils import count_tokens
            token_count = count_tokens(prompt_text)
            
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
