import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from embedding.embedder import SentenceTransformerEmbeddings, GeminiEmbeddings
from embedding.index_faiss import build_qdrant_index
from parsing.extract_text import extract_text
from retrieval.retriever import retrieve
from retrieval.rerank import rerank
from services.gemini import call_gemini
from parsing.chunking import semantic_chunking_with_st
import logging
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant

PDF_PATH = "./data/pdf/ifc-annual-report-2024-financials.pdf"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_resource
def load_qdrant():
    vectorstore = Qdrant(
        client=QdrantClient(host="localhost", port=6333),
        collection_name="ifc_report",
        embeddings=GeminiEmbeddings()
    )
    return vectorstore


# === Streamlit App ===
def main():
    st.title("IFC Annual Report RAG System")
    user_query = st.text_input("Ask a question about the report")
    with st.spinner("Loading and indexing document..."):
        vectorstore = load_qdrant()

    if user_query:
        with st.spinner("Processing query..."):
            relevant_chunks = retrieve(user_query, vectorstore)
            reranked_chunks = rerank(user_query, relevant_chunks, top_k=3)
            context = "\n\n".join([chunk.page_content for chunk in reranked_chunks])
            answer = call_gemini(context, user_query)

            st.markdown("### Answer:")
            st.write(answer)

            st.markdown("### Retrieved Context:")
            for chunk in reranked_chunks:
                st.markdown(f"> {chunk.page_content[:500]}...")


if __name__ == "__main__":
    main()
