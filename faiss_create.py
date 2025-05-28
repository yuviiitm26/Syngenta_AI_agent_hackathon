from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def build_faiss_index(pdf_dir="data/docs", index_path="faiss_index"):
    # Load and split documents
    documents = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, file))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Create embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create and save FAISS index
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(index_path)
    print("✅ FAISS index created and saved to:", index_path)

if __name__ == "__main__":
    build_faiss_index()
