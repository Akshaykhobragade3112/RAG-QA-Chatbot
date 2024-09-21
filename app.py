import streamlit as st
from rag_model import upload_embeddings, retrieve_documents, generate_answer
import PyPDF2

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text

# Streamlit UI
st.title("RAG-based QA Bot")

# Upload PDF documents
uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
documents = []

# If PDF is uploaded, extract text and store
if uploaded_files:
    for pdf_file in uploaded_files:
        pdf_text = extract_text_from_pdf(pdf_file)
        documents.append(pdf_text)
    
    st.write("Processing documents and uploading embeddings to FAISS...")
    upload_embeddings(documents)  # Upload document embeddings to FAISS
    st.write("Embeddings uploaded successfully!")

# Allow user to ask a question
question = st.text_input("Ask a question based on the uploaded documents:")

# If a question is entered, retrieve documents and generate an answer
if question:
    st.write(f"Your question: {question}")
    
    # Retrieve relevant documents from FAISS
    retrieved_docs = retrieve_documents(question)
    answer = generate_answer(question, retrieved_docs)
    
    # Display the generated answer
    st.write("Answer:")
    st.write(answer)
    
    # Optionally display retrieved document context (IDs)
    st.write("Retrieved documents:")
    for doc in retrieved_docs:
        st.write(f"Document content: {doc}")
