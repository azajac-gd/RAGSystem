import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain.vectorstores import Qdrant
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
from parsing.chunking import semantic_chunking_with_st
from embedding.embedder import GeminiEmbeddings
from services.gemini import summarize_image
from langchain.schema import Document
from parsing.docling_images import main
import json
import pandas as pd
from pathlib import Path


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


def add_images_to_index(vectorstore, figures_json_path: str):
    print("Adding images to index from JSON...")

    all_chunks = []

    with open(figures_json_path, "r") as f:
        image_metadata = json.load(f)

    for image_obj in image_metadata:
        try:
            image_path = Path(image_obj["image_path"])
            with image_path.open("rb") as img_file:
                image_bytes = img_file.read()

            summary = summarize_image(image_bytes, image_obj["title"])

            doc = Document(
                page_content=summary,
                metadata={
                    "title": image_obj.get("title", "Unknown"),
                    "page": image_obj.get("page_number", -1),
                    "section": image_obj.get("section_title", "Unknown"),
                    "type": "image",
                    "content": image_obj.get("content", "Unknown"),
                    "image_path": str(image_path),
                }
            )
            all_chunks.append(doc)

        except Exception as e:
            print(f"Error on image: {image_obj.get('image_path')}: {e}")

    vectorstore.add_documents(all_chunks)
    print(f"Added {len(all_chunks)} image summaries to the index.")


def load_and_index_documents():
    print("Extracting text from PDF...")
    docs = main()
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

def chunk_tables_from_csv_and_metadata(vectorstore, csv_dir, metadata_path):
    with open(metadata_path, "r") as f:
        table_metadata = json.load(f)

    all_documents = []

    for meta in table_metadata:
        csv_file = Path(csv_dir) / meta["csv"]
        if not csv_file.exists():
            print(f"Warning: {csv_file} not found.")
            continue

        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue

        title = meta.get("title", "Unknown")
        page = meta.get("page_number", -1)
        section = meta.get("section", "Unknown")
        content_title = meta.get("content", "Unknown")

        headers = list(df.columns)
        for idx, row in df.iterrows():
            content = f"Title: {title}\n" \
                      f"Table row:\n" \
                      f"{' | '.join(headers)}\n" \
                      f"{' | '.join(str(row[col]) for col in headers)}"
            doc = Document(
                page_content=content,
                metadata={
                    "page": page,
                    "section": section,
                    "type": "table",
                    "row_index": idx,
                    "title": title,
                    "content": content_title
                }
            )
            all_documents.append(doc)

    vectorstore.add_documents(all_documents)
    print(f"Added {len(all_documents)} tables to the index.")



#load_and_index_documents()
#add_images_to_index(load_qdrant(), "./scratch/ifc-annual-report-2024-financials-figures.json")
#chunk_tables_from_csv_and_metadata(load_qdrant(), "./", "./scratch/ifc-annual-report-2024-financials-tables.json")
