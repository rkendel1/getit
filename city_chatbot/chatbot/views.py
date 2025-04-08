from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from transformers import pipeline
from .vector_store import vector_store
import os
import json
from django.views.decorators.csrf import csrf_exempt


def scrape_url(request):
    if request.method == 'POST':
        url = request.POST.get('url')
        try:
            vector_store.add_scraped_data(url)
            return render(request, 'scrape.html', {'message': 'Scraping successful.'})
        except Exception as e:
            return render(request, 'scrape.html', {'error': f'Error: {e}'})
    return render(request, 'scrape.html')


def upload_pdf(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('pdf_file')
        if uploaded_file and uploaded_file.name.endswith('.pdf'):
            
            file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            vector_store.add_pdf(file_path)  # This will handle chunking & indexing
            return redirect('upload_pdf')  # or a 'success' view
        else:
            return render(request, 'upload.html', {'error': 'Invalid file. Please upload a PDF.'})
    return render(request, 'upload.html')


# Load QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def chatbot_view(request):
    return render(request, 'chatbot/chatbot.html')


@csrf_exempt
def chat_response(request):
    if request.method == 'POST':
        try:
            # Check if request is JSON or form data and parse accordingly
            if request.content_type == 'application/json':
                data = json.loads(request.body)
                user_input = data.get('question', '').strip()
            else:
                user_input = request.POST.get('message', '').strip()
        except Exception as e:
            return JsonResponse({'response': f'Error parsing input: {str(e)}'})

        if not user_input:
            return JsonResponse({'response': "Please ask me something about my city!"})

        # Ensure the vector store is initialized and contains data
        if not vector_store or not vector_store.index:
            return JsonResponse({'response': "No data available. Upload a PDF or scrape a site first."})

        # Retrieve relevant context from the vector store based on the user's input
        contexts = vector_store.search(user_input, k=3)
        context = ' '.join(contexts)
        if not context:
            return JsonResponse({'response': "I don’t have enough info yet—upload a PDF or scrape a site!"})

        # Generate response using the question-answering pipeline
        result = qa_pipeline({'question': user_input, 'context': context})
        response = result['answer'] if result['score'] > 0.2 else "I’m not sure, but here’s what I know: " + context[:200] + "..."
        
        return JsonResponse({'response': response})

    return JsonResponse({'response': "Invalid request"})