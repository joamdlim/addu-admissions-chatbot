from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat),
    path('upload/', views.upload),
    path('evaluate/', views.evaluate),
]
