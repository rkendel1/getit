from django.urls import path
from . import views
from chatbot import views
from django.conf import settings
from django.conf.urls.static import static



urlpatterns = [
    path('', views.chatbot_view, name='chatbot'),
    path('chat/', views.chat_response, name='ask_question'),  # ✅ make sure this name is used in HTML
    path('upload/', views.upload_pdf, name='upload_pdf'),
    path('scrape/', views.scrape_url, name='scrape_url'),

   
]

 
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)  