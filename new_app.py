import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Page configuration
st.set_page_config(page_title="C++ RAG ChatBot", page_icon="💬", layout="wide")
st.title("C++ RAG ChatBot 💬")
st.write("Ask any question related to C++ and get the answer")

# Step 2: Load environment variables
load_dotenv()

# Step 3: Cache the document loading and processing
@st.cache_resource
def load_vector_store():
    # Step A: Load documents
    loader = TextLoader("Cpp.txt", encoding="utf-8")
    documents = loader.load()

    # Step B: Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    final_documents = text_splitter.split_documents(documents)

    # Step C: Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step D: Create FAISS vector store
    db = FAISS.from_documents(final_documents, embeddings)
    return db

# Vector database runs only once because of caching
db = load_vector_store()

# User Input
user_query = st.text_input("Enter your question about C++:")

if user_query:
    # Convert user question to embeddings and search FAISS
    documents = db.similarity_search(user_query, k=3)

    st.subheader("📖 Retrieved context:")

    for i, doc in enumerate(documents):
        st.markdown(f"*Result  {i+1}:*")
        st.write(doc.page_content)