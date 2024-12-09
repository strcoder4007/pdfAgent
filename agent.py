import os
from dotenv import load_dotenv
import streamlit as st
from langchain_ollama import OllamaLLM
import fitz
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import torch

# Load environment variables
load_dotenv()

# Initialize LLM
llm = OllamaLLM(model="llama3.1:latest", base_url="http://localhost:11434")

# Load the E5 model for embeddings on CPU
model_name = 'intfloat/e5-large-v2'
device = 'cpu'  # Force using CPU
model = SentenceTransformer(model_name, device=device)

# Initialize FAISS index and chunks list
dim = model.get_sentence_embedding_dimension()
faiss_index = faiss.IndexHNSWFlat(dim, 32)
chunks_list = []

# Load existing index and chunks if available
index_exists = os.path.exists('faiss_index.bin')
chunks_exists = os.path.exists('chunks.pkl')

if index_exists and chunks_exists:
    faiss_index = faiss.read_index('faiss_index.bin')
    with open('chunks.pkl', 'rb') as f:
        chunks_list = pickle.load(f)

# Streamlit app title
st.title("PDF Question Answering")

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        text += doc.load_page(page_num).get_text()
    return text

def chunk_text(text, max_length=500):
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        if len(' '.join(current_chunk) + ' ' + word) > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def generate_embeddings(texts):
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings.numpy().astype(np.float32)

def upsert_to_faiss(chunks, embeddings):
    global faiss_index, chunks_list
    num_vectors = len(chunks)
    if faiss_index.ntotal == 0:
        faiss_index = faiss.IndexHNSWFlat(dim, 32)
    faiss_index.add(embeddings)
    chunks_list.extend(chunks)
    # Save the index and chunks
    faiss.write_index(faiss_index, 'faiss_index.bin')
    with open('chunks.pkl', 'wb') as f:
        pickle.dump(chunks_list, f)

def query_faiss(question, top_k=3):
    query_embedding = model.encode([question], convert_to_tensor=True).numpy().astype(np.float32)
    D, I = faiss_index.search(query_embedding, top_k)
    context = [chunks_list[i] for i in I[0]]
    return context, D[0]

def get_answer(question, context, llm):
    system = "Give your answers in 1-2 sentences."
    prompt = f"System: {system}\nContext: {context}\nQuestion: {question}\nAnswer:"
    response = llm.invoke(prompt)
    return response.strip()

def main():
    # File uploader in Streamlit
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Save the uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract text from the PDF
            text = extract_text_from_pdf("temp.pdf")
            
            # Chunk the text
            chunks = chunk_text(text)
            
            # Generate embeddings for chunks
            embeddings = generate_embeddings(chunks)
            
            # Upsert to FAISS
            upsert_to_faiss(chunks, embeddings)
            st.write("File embeddings saved!")
    
    # Text area for questions
    questions_input = st.text_area("Enter your questions (one per line):")
    questions = [q.strip() for q in questions_input.split('\n') if q.strip()]
    
    if st.button("Get Answers"):
        if not questions:
            st.warning("Please enter at least one question.")
            return
        
        if faiss_index.ntotal == 0:
            st.warning("Please upload a PDF file and build the index first.")
            return
        
        with st.spinner("Generating answers..."):
            answers = {}
            for q in questions:
                # Retrieve relevant context
                context, distances = query_faiss(q)
                context_str = ' '.join(context)
                
                # Confidence threshold (placeholder value)
                confidence_threshold = 1.5  # Adjust as needed
                if np.min(distances) > confidence_threshold:
                    answers[q] = "Data Not Available"
                else:
                    # Get answer from LLM
                    answer = get_answer(q, context_str, llm)
                    answers[q] = answer
            
            # Display answers as JSON
            st.json(answers)
            
            # Allow user to download the JSON
            json_str = json.dumps(answers, indent=4)
            st.download_button(
                label="Download Answers as JSON",
                data=json_str,
                file_name="answers.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()