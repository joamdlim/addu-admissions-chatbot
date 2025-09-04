from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat_view, name='chat'),
    path("upload-pdf/", views.upload_pdf_view, name="upload_pdf"),
    path("sync-supabase/", views.sync_supabase_to_chroma, name="sync_supabase"),
    path("export-chroma/", views.export_chroma_to_local, name="export_chroma"),
    
    # Admin endpoints
    path("admin/upload/", views.admin_upload_file, name="admin_upload"),
    path("admin/files/", views.admin_list_files, name="admin_list_files"),
    path("admin/delete/", views.admin_delete_file, name="admin_delete_file"),
    
    # Keep only the sync endpoint for manual syncing
    path("admin/sync-to-chroma/", views.admin_sync_file_to_chroma, name="admin_sync_to_chroma"),
]
