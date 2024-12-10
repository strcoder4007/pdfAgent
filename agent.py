import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import fitz
import numpy as np
import json
import tiktoken
from sklearn.preprocessing import normalize
import time

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0.2
)

# Load the E5 model for embeddings on CPU
model_name = 'intfloat/e5-large-v2'
device = 'cpu'
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
    """
    Extracts text from a PDF file.
    """
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        text += doc.load_page(page_num).get_text()
    return text

def chunk_text(text, max_length=500):
    """
    Splits the input text into chunks of a specified maximum length.
    """
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
    """
    Generates and normalizes embeddings for a list of texts.
    """
    embeddings = model.encode(texts, convert_to_tensor=True).numpy().astype(np.float32)
    embeddings = normalize(embeddings, norm='l2')
    return embeddings

def upsert_to_faiss(chunks, embeddings):
    """
    Adds normalized chunks and embeddings to the FAISS index.
    """
    global faiss_index, chunks_list
    if faiss_index.ntotal == 0:
        faiss_index = faiss.IndexHNSWFlat(dim, 32)
    faiss_index.add(embeddings)
    chunks_list.extend(chunks)
    faiss.write_index(faiss_index, 'faiss_index.bin')
    with open('chunks.pkl', 'wb') as f:
        pickle.dump(chunks_list, f)

def query_faiss(question, top_k=3):
    """
    Queries the FAISS index for top_k similar chunks.
    """
    query_embedding = model.encode([question], convert_to_tensor=True).numpy().astype(np.float32)
    query_embedding = normalize(query_embedding, norm='l2')
    D, I = faiss_index.search(query_embedding, top_k)
    context = [chunks_list[i] for i in I[0]]
    return context, D[0]

def count_tokens(text):
    """
    Counts the number of tokens in the input text.
    """
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))

def get_answer(question, context, llm):
    """
    Generates an answer based on the provided context.
    """
    system = (
        "You are an assistant designed to answer questions based solely on the provided context. "
        "If the answer is directly available in the context, provide it verbatim. Otherwise, summarize the relevant information concisely, ideally in 2-3 sentences."
    )
    context_str = ' '.join(context)
    if count_tokens(context_str) > 1200:
        context_str = ' '.join(context_str.split()[:1000])
    if not context_str:
        context_str = "No relevant context found."
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Given the following context: {context_str}. Answer the question: {question}.")
    ]
    response = llm.invoke(messages)
    return response.content.strip()

def main():
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None and not st.session_state.file_uploaded:
        start_time = time.time()
        with st.spinner("Processing PDF... (It might take 4-5 mins for long pdfs to process on CPU)"):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            text = extract_text_from_pdf("temp.pdf")
            chunks = chunk_text(text)
            embeddings = generate_embeddings(chunks)
            upsert_to_faiss(chunks, embeddings)
            elapsed_time = time.time() - start_time
            st.write(f"File embeddings saved! Processing time: {elapsed_time:.1f} seconds")
        st.session_state.file_uploaded = True
        os.remove("temp.pdf")
    
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
                context, distances = query_faiss(q)
                context_str = ' '.join(context)
                confidence_threshold = 0.45
                if np.min(distances) > confidence_threshold:
                    answers[q] = "Data Not Available"
                else:
                    answer = get_answer(q, context_str, llm)
                    answers[q] = answer
            st.json(answers)

if __name__ == "__main__":
    main()