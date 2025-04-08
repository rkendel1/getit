import os
import glob

for file in glob.glob("data/index_*.faiss"):
    os.remove(file)
    
#os.remove("chatbot_data/vector_index.pkl")