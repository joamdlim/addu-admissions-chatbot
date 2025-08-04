from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat_view, name='chat'),
    path("chroma/test-add/", views.chroma_test_add, name="chroma_test_add"),
    path("chroma/test-query/", views.chroma_test_query, name="chroma_test_query"),
]
