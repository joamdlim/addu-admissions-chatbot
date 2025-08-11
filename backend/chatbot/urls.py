from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat_view, name='chat'),
    path("chroma/test-add/", views.chroma_test_add, name="chroma_test_add"),
    path("chroma/test-query/", views.chroma_test_query, name="chroma_test_query"),
    path("upload-pdf/", views.upload_pdf_view, name="upload_pdf"),
    path("sync-supabase/", views.sync_supabase_to_chroma, name="sync_supabase"),
    path("export-chroma/", views.export_chroma_to_local, name="export_chroma"),
]
