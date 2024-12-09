import os
from dotenv import load_dotenv
import streamlit as st
from langchain_ollama import OllamaLLM
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import fitz
import numpy as np
import json

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("first")

# Initialize LLM
llm = OllamaLLM(model="llama3.1:latest", base_url="http://localhost:11434")

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
    
    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks



def generate_embeddings(texts, model_name="multilingual-e5-large"):
    embeddings = pc.inference.embed(
        model=model_name,
        inputs=texts,
        parameters={"input_type": "passage", "truncate": "END"}
    )
    return [e['values'] for e in embeddings]




def upsert_to_pinecone(chunks, embeddings):
    """Upsert chunks and embeddings to Pinecone."""
    records = [
        {
            "id": f"doc_{i}",
            "values": embedding,
            "metadata": {"text": chunk}
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    index.upsert(vectors=records, namespace="default")



def query_pinecone(question, top_k=3, model_name="multilingual-e5-large"):
    """Query Pinecone for top K chunks similar to the question."""
    query_embedding = pc.inference.embed(
        model=model_name,
        inputs=[question],
        parameters={"input_type": "query"}
    )[0]['values']
    
    results = index.query(
        namespace="default",
        vector=query_embedding,
        top_k=top_k,
        include_values=False,
        include_metadata=True
    )
    return [match.metadata['text'] for match in results.matches]



def get_answer(question, context, llm):
    system = "Give your answers in 1-2 sentences."
    prompt = f"System: {system}\nContext: {context}\nQuestion: {question}\nAnswer:"
    response = llm(prompt)
    return response.strip()




def main():
    # File uploader in Streamlit
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text from the PDF
        text = extract_text_from_pdf("temp.pdf")
        
        # Chunk the text
        chunks = chunk_text(text)
        
        # Generate embeddings for chunks
        embeddings = generate_embeddings(chunks)
        
        # Upsert to Pinecone
        upsert_to_pinecone(chunks, embeddings)
        st.write("File embeddings saved!")
    
    # Text area for questions
    questions_input = st.text_area("Enter your questions (one per line):")
    questions = [q.strip() for q in questions_input.split('\n') if q.strip()]
    
    if st.button("Get Answers"):
        if not questions:
            st.warning("Please enter at least one question.")
            return
        
        if "temp.pdf" not in os.listdir():
            st.warning("Please upload a PDF file first.")
            return
        
        answers = {}
        for q in questions:
            # Retrieve relevant context
            context = query_pinecone(q)
            context_str = ' '.join(context)
            
            # Get answer from LLM
            answer = get_answer(q, context_str, llm)
            
            # Placeholder for confidence check
            # For demonstration, assume "Data Not Available" if context is empty
            if not context_str:
                answers[q] = "Data Not Available"
            else:
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