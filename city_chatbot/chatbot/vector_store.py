from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import PyPDF2
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json


class VectorStore:
    def __init__(self, data_dir):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Free embedding model
        self.index = None
        self.sentences = []
        self.data_dir = data_dir
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.load_existing_data()
        self.build_index()

    def load_existing_data(self):
        text_file = os.path.join(self.data_dir, 'processed_data.txt')
        if os.path.exists(text_file):
            with open(text_file, 'r') as f:
                self.sentences = [line.strip() for line in f if line.strip()]
            print(f"✅ Loaded {len(self.sentences)} sentences from processed_data.txt")
        else:
            print("⚠️ No processed_data.txt found. Starting fresh.")

    def build_index(self):
        # Example implementation: Build a FAISS index
        if self.sentences:
            print("Building FAISS index...")
            embeddings = self.model.encode(self.sentences)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings))
            print("Index built successfully.")
        else:
            print("No sentences available to build the index.")

    def add_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''
            
        chunks = [chunk.strip() for chunk in self.splitter.split_text(text) if chunk.strip()]
        self.sentences.extend(chunks)
        self.save_and_rebuild()

    def add_scraped_data(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        chunks = [chunk.strip() for chunk in self.splitter.split_text(text) if chunk.strip()]
        self.sentences.extend(chunks)
        # Save structured data (e.g., title) as JSON
        structured_data = {'url': url, 'title': soup.title.string if soup.title else 'No title'}
        with open(os.path.join(self.data_dir, 'scraped_structured.json'), 'a') as f:
            f.write(f"{json.dumps(structured_data)}\n")
        self.save_and_rebuild()

    def save_and_rebuild(self):
        with open(os.path.join(self.data_dir, 'processed_data.txt'), 'w') as f:
            f.write('\n'.join(self.sentences))
        embeddings = self.model.encode(self.sentences, convert_to_tensor=False)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))
        faiss.write_index(self.index, os.path.join(self.data_dir, 'index.faiss'))

    def search(self, query, k=3):
        if not self.index or self.index.ntotal == 0:
            return ["Sorry, I don’t have enough data to answer that right now."]

        try:
            query_embedding = self.model.encode([query], convert_to_tensor=False)
            distances, indices = self.index.search(query_embedding.astype(np.float32), k)
            hits = [self.sentences[idx] for idx in indices[0] if idx < len(self.sentences)]
            if not hits:
                return ["I couldn’t find a relevant answer. Try rephrasing your question."]
            return hits
        except Exception as e:
            print("🔴 Search error:", str(e))
            return [f"An error occurred during search: {str(e)}"]

from django.conf import settings

data_dir = getattr(settings, 'DATA_DIR', 'chatbot_data')  # fallback just in case

if data_dir:
    vector_store = VectorStore(data_dir)
    faiss_index_path = os.path.join(data_dir, 'index.faiss')
    if os.path.exists(faiss_index_path):
        print("✅ Loading FAISS index from disk...")
        vector_store.index = faiss.read_index(faiss_index_path)
    else:
        print("⚠️ FAISS index not found. Will build after adding data.")
else:
    print("❌ DATA_DIR not set in settings")
    vector_store = None