import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain.vectorstores import Qdrant
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
from parsing.extract_text import extract_text, extract_text_with_docling
from parsing.chunking import semantic_chunking_with_st
from embedding.embedder import GeminiEmbeddings

load_dotenv()
PDF_PATH = "./data/pdf/ifc-annual-report-2024-financials.pdf"
QDRANT_COLLECTION = "ifc_report"


@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    return QdrantClient(host="localhost", port=6333)


def build_qdrant_index(chunks, embedding_model, collection_name="my_collection"):
    client = get_qdrant_client()

    if collection_name in [c.name for c in client.get_collections().collections]:
        client.delete_collection(collection_name=collection_name)

    vector_size = len(embedding_model.embed_query("sample text"))

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    vectorstore = Qdrant.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        host="localhost",
        port=6333
    )

    return vectorstore


def load_and_index_documents():
    docs = extract_text(PDF_PATH)
    full_text = " ".join([doc.page_content for doc in docs])
    documents = semantic_chunking_with_st(full_text)
    embedding_model = GeminiEmbeddings()
    vectorstore = build_qdrant_index(
        documents,
        embedding_model,
        collection_name=QDRANT_COLLECTION
    )
    return vectorstore

#load_and_index_documents()
