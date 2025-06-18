import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from embedding.embedder import SentenceTransformerEmbeddings, GeminiEmbeddings
from embedding.index_faiss import build_qdrant_index
from parsing.extract_text import extract_text
from retrieval.retriever import retrieve
from retrieval.rerank import rerank
from services.gemini import call_gemini, return_metadata
from parsing.chunking import semantic_chunking_with_st
import logging
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from embedding.index_faiss import get_qdrant_client

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
            metadata = return_metadata(user_query)
            query, page_start, page_end, section, _type  = metadata
            if metadata:
                st.markdown("### Metadata:")
                st.json(metadata)
            relevant_chunks = retrieve(vectorstore, query, page_start, page_end, section, _type)
            st.write(f"Found {len(relevant_chunks)} relevant chunks.")
            if len(relevant_chunks) == 0:
                st.error("No relevant chunks found. Please try a different query.")
                return
            else:
                reranked_chunks = rerank(user_query, relevant_chunks, top_k=3)
                context = "\n\n".join([chunk.page_content for chunk in reranked_chunks])
                answer = call_gemini(context, user_query)

                if answer != "The document does not provide this information." and answer != "This question is not related to the IFC Annual Report 2024." and answer != "The answer is not in the provided pages.":
                    st.markdown("### Answer:")
                    st.write(answer)
                    st.markdown("### Retrieved Context:")
                    for chunk in reranked_chunks:
                        st.markdown(f"> {chunk.page_content[:500]}...")
                else:
                    st.error(answer)

if __name__ == "__main__":
    main()
