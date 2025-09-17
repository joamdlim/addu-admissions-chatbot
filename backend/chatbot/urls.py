from django.urls import path
from . import views

urlpatterns = [
    # Main chat endpoints
    path('chat/', views.chat_view, name='chat'),  # Enhanced chat with conversation memory
    path('chat/legacy/', views.chat_legacy_view, name='chat_legacy'),  # Backward compatibility
    
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
]
