from django.urls import path
from . import views

urlpatterns = [
    # Main chat endpoint - guided chatbot only
    path('chat/guided/', views.guided_chat_view, name='guided_chat'),  # Guided conversation with topics
    
    # Topics endpoint
    path('topics/', views.get_topics, name='get_topics'),
    
    # Conversation management endpoints
    path('conversations/', views.create_conversation, name='create_conversation'),
    path('conversations/active/', views.list_active_conversations, name='list_active_conversations'),
    path('conversations/<str:session_id>/history/', views.get_conversation_history, name='get_conversation_history'),
    path('conversations/<str:session_id>/stats/', views.get_conversation_stats, name='get_conversation_stats'),
    path('conversations/<str:session_id>/', views.clear_conversation, name='clear_conversation'),
    path('conversations/<str:session_id>/summarize/', views.force_summarization, name='force_summarization'),
    
    # System prompt management
    path('system-prompts/', views.get_system_prompts, name='get_system_prompts'),
    path('system-prompts/update/', views.update_system_prompt, name='update_system_prompt'),
    
    # Legacy endpoints (keeping for backward compatibility)
    path("upload-pdf/", views.upload_pdf_view, name="upload_pdf"),
    path("sync-supabase/", views.sync_supabase_to_chroma, name="sync_supabase"),
    path("export-chroma/", views.export_chroma_to_local, name="export_chroma"),
    
    # Admin endpoints
    path("admin/upload/", views.admin_upload_file, name="admin_upload"),
    path("admin/files/", views.admin_list_files, name="admin_list_files"),
    path("admin/delete/", views.admin_delete_file, name="admin_delete_file"),
    path("admin/sync-to-chroma/", views.admin_sync_file_to_chroma, name="admin_sync_to_chroma"),
    path("admin/extract-text/", views.extract_text_from_file, name="extract_text_from_file"),
    path("admin/upload-processed/", views.upload_processed_file, name="upload_processed_file"),
    path("admin/download/<str:file_name>/", views.admin_download_file, name="admin_download_file"),
    
    # Folder management endpoints
    path("admin/folders/", views.manage_folders, name="manage_folders"),
    path("admin/folders/all/", views.get_all_folders, name="get_all_folders"),
    path("admin/folders/<int:folder_id>/", views.manage_folder_detail, name="manage_folder_detail"),
    path("admin/folder-tree/", views.get_folder_tree, name="get_folder_tree"),
    
    # Document metadata management
    path("admin/documents/", views.manage_document_metadata, name="manage_document_metadata"),
    path("admin/documents/<str:document_id>/metadata/", views.update_document_metadata, name="update_document_metadata"),
    path("admin/documents/<str:document_id>/", views.delete_document, name="delete_document"),
    
    # Enhanced chat with filtering
    path("chat/enhanced/", views.enhanced_chat_query, name="enhanced_chat_query"),
    
    # Topic management endpoints
    path("admin/topics/", views.get_topics_admin, name="get_topics_admin"),
    path("admin/topics/<str:topic_id>/keywords/", views.get_topic_keywords, name="get_topic_keywords"),
    path("admin/topics/<str:topic_id>/keywords/add/", views.add_topic_keyword, name="add_topic_keyword"),
    path("admin/topics/<str:topic_id>/keywords/bulk/", views.bulk_add_topic_keywords, name="bulk_add_topic_keywords"),
    path("admin/keywords/<int:keyword_id>/update/", views.update_keyword, name="update_keyword"),
    path("admin/keywords/<int:keyword_id>/delete/", views.delete_keyword, name="delete_keyword"),
]
