import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import time
from openai.error import RateLimitError

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY", "<Your OpenAI Secret Key>")

# Initialize FAISS Index (for 768-dimensional vectors)
embedding_dim = 768
index = faiss.IndexFlatL2(embedding_dim)

# Load the sentence transformer model for embedding generation (768-dimensional)
embedder = SentenceTransformer('all-mpnet-base-v2')

# Dictionary to store document embeddings with their IDs
documents_store = {}

# Function to upload document embeddings to FAISS
def upload_embeddings(documents):
    global documents_store
    
    embeddings = embedder.encode(documents)  # Convert documents to embeddings
    ids = [f"doc-{i}" for i in range(len(documents))]  # Generate document IDs
    
    # Add embeddings to FAISS index
    index.add(np.array(embeddings))
    
    # Store embeddings along with IDs
    for i, doc_id in enumerate(ids):
        documents_store[doc_id] = documents[i]

# Function to retrieve relevant documents based on a query
def retrieve_documents(query, top_k=5):
    query_embedding = embedder.encode([query])[0]  # Embed the query
    
    # Check if there are documents in FAISS
    if index.ntotal == 0:
        return []

    # Search the FAISS index for similar embeddings
    distances, indices = index.search(np.array([query_embedding]), top_k)
    
    # Make sure indices are within bounds and retrieve corresponding documents
    retrieved_docs = []
    for idx in indices[0]:
        if idx < len(documents_store):
            retrieved_docs.append(list(documents_store.values())[idx])

    return retrieved_docs

# Function to generate an answer using GPT-3.5-turbo and the retrieved context
def generate_answer(question, retrieved_docs, retries=3, backoff_factor=2):
    if not retrieved_docs:
        return "No relevant documents found."
    
    context = "\n".join(retrieved_docs)  # Combine the retrieved documents into context
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Question: {question}\nContext: {context}"}
    ]

    attempt = 0
    while attempt < retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Use the new GPT-3.5-turbo model
                messages=messages,
                max_tokens=200,
                temperature=0.7  # Adjust creativity level
            )
            return response['choices'][0]['message']['content'].strip()

        except RateLimitError as e:
            attempt += 1
            wait_time = backoff_factor ** attempt
            print(f"RateLimitError: Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    return "API request failed due to rate limits. Please try again later."

# Example usage (for testing purposes)
if __name__ == "__main__":
    documents = ["Sample document content here.", "Another document to embed."]
    upload_embeddings(documents)  # Upload the documents
    
    question = "What is the content of the first document?"
    retrieved_docs = retrieve_documents(question)  # Retrieve relevant docs
    answer = generate_answer(question, retrieved_docs)  # Get GPT-3 generated answer
    
    print("Answer:", answer)
