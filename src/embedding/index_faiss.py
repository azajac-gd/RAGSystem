from langchain.vectorstores import Qdrant
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from embedding.embedder import (
    save_embeddings_cache,
    load_embeddings_cache,
    SentenceTransformerEmbeddings,
    GeminiEmbeddings,
)
from dotenv import load_dotenv

load_dotenv()
PDF_PATH = "./data/pdf/ifc-annual-report-2024-financials.pdf"
QDRANT_COLLECTION = "ifc_report"
EMBEDDINGS_CACHE_PATH = "./embedding_cache.json"

@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    return QdrantClient(host="localhost", port=6333)

def build_qdrant_index(chunks, embedding_model):
    qdrant_client = get_qdrant_client()

    collections = [c.name for c in qdrant_client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        qdrant_client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        st.info("Created new Qdrant collection.")
    else:
        st.info("Using existing Qdrant collection.")

    embeddings = load_embeddings_cache(EMBEDDINGS_CACHE_PATH)
    if embeddings is None or len(embeddings) != len(chunks):
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_model.embed_documents(texts)
        save_embeddings_cache(EMBEDDINGS_CACHE_PATH, embeddings)
    else:
        st.info("Loaded embeddings from cache.")

    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]

        points = []
        for idx, (chunk, emb) in enumerate(zip(batch_chunks, batch_embeddings)):
            points.append({
                "id": i + idx,  
                "vector": emb,
                "payload": chunk.metadata or {}
            })

        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )
        st.write(f"Upserted batch {i//batch_size + 1} / {(len(chunks) + batch_size - 1)//batch_size}")

    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION,
        embedding=embedding_model,
    )
    return vectorstore

def build_qdrant_index_with_sentence_transformer(chunks, embedding_model, collection_name="my_collection"):
    client = QdrantClient(host="localhost", port=6333)

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
import faiss
from langchain.vectorstores import FAISS

def build_faiss_index(chunks, embedding_model):
    st.info("Building FAISS index...")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.embed_documents(texts) 
    st.info(f"Generated {len(embeddings)} embeddings.")
    import numpy as np
    embeddings_np = np.array(embeddings).astype('float32')
    st.info(f"Converted embeddings to numpy array with shape {embeddings_np.shape}.")
    vectorstore = FAISS.from_documents(chunks, embedding_model, index=faiss.IndexFlatIP(embeddings_np.shape[1]))

    vectorstore.index.add(embeddings_np)

    return vectorstore

def build_faiss_index_with_sentence_transformer(chunks, embedding_model):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore