from langchain.vectorstores import Qdrant
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import faiss
from langchain.vectorstores import FAISS

load_dotenv()
PDF_PATH = "./data/pdf/ifc-annual-report-2024-financials.pdf"
QDRANT_COLLECTION = "ifc_report"

@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    return QdrantClient(host="localhost", port=6333)

def build_qdrant_index(chunks, embedding_model, collection_name="my_collection"):
    client = QdrantClient(host="localhost", port=6333) #to sprobowac zastapic 

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




def build_faiss_index_with_sentence_transformer(chunks, embedding_model):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore