import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain.vectorstores import Qdrant
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
from parsing.extract_text import extract_text
from parsing.chunking import semantic_chunking_with_st
from embedding.embedder import GeminiEmbeddings
from parsing.extract_images import extract_images_from_pdf
from services.gemini import summarize_image
from langchain.schema import Document

load_dotenv()
PDF_PATH = "./data/pdf/ifc-annual-report-2024-financials.pdf"
QDRANT_COLLECTION = "ifc_report"

def load_qdrant():
    vectorstore = Qdrant(
        client=QdrantClient(host="localhost", port=6333),
        collection_name="ifc_report",
        embeddings=GeminiEmbeddings()
    )
    return vectorstore

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

def add_images_to_index(vectorstore):
    print("Adding images to index...")
    embedding_model = GeminiEmbeddings()
    image_objects = extract_images_from_pdf(PDF_PATH)

    all_chunks = []
    for image_obj in image_objects:
        try:
            summary = summarize_image(image_obj['image_bytes'])
            doc = Document(
                page_content=summary,
                metadata={
                    "page": image_obj['page'],
                    "section": None,
                    "type": "image",
                }
            )
            all_chunks.append(doc)
        except Exception as e:
            print(f"Error on page: {image_obj['page']}: {e}")

    vectorstore.add_documents(all_chunks)
    print(f"Added {len(all_chunks)} image summaries to the index.")


def load_and_index_documents():
    print("Extracting text from PDF...")
    docs = extract_text(PDF_PATH)
    print(f"Extracted {len(docs)} text documents.")
    embedding_model = GeminiEmbeddings()

    all_chunks = []
    print("Chunking documents...")
    for doc in docs:
        chunks = semantic_chunking_with_st(doc.page_content)
        for chunk in chunks:
            chunk.metadata = doc.metadata
            all_chunks.append(chunk)
    print(f"Created {len(all_chunks)} chunks from text documents.")

    vectorstore = build_qdrant_index(
        all_chunks,
        embedding_model,
        collection_name=QDRANT_COLLECTION
    )
    return vectorstore


##load_and_index_documents()
add_images_to_index(load_qdrant())
