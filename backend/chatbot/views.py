from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from fast_hybrid_chatbot import FastHybridChatbot
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
from .chroma_connection import ChromaService
from .test_pdf_to_chroma import process_and_store_pdf_in_chroma

# Initialize the chatbot
chatbot = FastHybridChatbot()

# @api_view(['POST'])
# def chat(request):
#     user_input = request.data.get("message")
    
#     if not user_input:
#         return Response({"error": "No message provided"}, status=400)
    
#     # Process the query using our hybrid chatbot
#     response, relevant_docs = chatbot.process_query(user_input)
    
#     # Format the response for the frontend
#     result = {
#         "reply": response,
#         "sources": [
#             {
#                 "id": doc["id"],
#                 "content": doc["content"][:100] + "..." if len(doc["content"]) > 100 else doc["content"],
#                 "relevance": f"{doc['relevance']:.2f}"
#             }
#             for doc in relevant_docs
#         ]
#     }
    
#     return Response(result)

# @api_view(['POST'])
# def upload(request):
#     # TODO: Implement document upload, preprocessing, and vectorization
#     return Response({"status": "uploaded"})

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
        data = json.loads(request.body)
        prompt = data.get("prompt", "")
        response, _ = chatbot.process_query(prompt)
        return JsonResponse({"response": response})
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
