import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from embedding.embedder import GeminiEmbeddings
from retrieval.retriever import retrieve, search_with_scores
from retrieval.rerank import rerank
from services.gemini import call_gemini, return_metadata
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
def main(rerank_enabled=True):
    st.title("IFC Annual Report RAG System")
    user_query = st.text_input("Ask a question about the report")
    with st.spinner("Loading and indexing document..."):
        vectorstore = load_qdrant()

    if user_query:
        with st.spinner("Processing query..."):
            metadata = return_metadata(user_query)
            logging.info(f"Metadata returned: {metadata}")
            query, page_start, page_end, _type  = metadata
            relevant_chunks = retrieve(vectorstore, query, page_start, page_end, _type)
            search_with_scores(vectorstore.client, vectorstore.embeddings, query, vectorstore.collection_name, page_start, page_end, _type)
            if len(relevant_chunks) == 0:
                st.error("No relevant chunks found. Please try a different query.")
                return
            else:
                if rerank_enabled:
                    reranked_chunks = rerank(user_query, relevant_chunks, top_k=10)
                    relevant_chunks = reranked_chunks
                answer = call_gemini(relevant_chunks, user_query)

                if answer != "The document does not provide this information." and answer != "This question is not related to the IFC Annual Report 2024." and answer != "The answer is not in the provided pages.":
                    st.markdown("### Answer:")
                    st.write(answer)
                    st.markdown("### Retrieved Context")
                    with st.expander("Show retrieved context chunks"):
                        for i, chunk in enumerate(relevant_chunks):
                            page = chunk.metadata.get("page", "N/A")
                            section = chunk.metadata.get("content", "Unknown")
                            section_1 = chunk.metadata.get("section", None)
                            content_type = chunk.metadata.get("type", "text")
                            if section_1 != "Unknown":
                                st.markdown(f""" 
                                #### Chunk {i+1}
                                - **Section:** {section}: {section_1}  
                                - **Page:** {page}  
                                - **Type:** `{content_type}`  
                                """)
                                st.write(chunk.page_content[:500] + "...")
                            else:
                                st.markdown(f""" 
                                #### Chunk {i+1}
                                - **Section:** {section} 
                                - **Page:** {page}  
                                - **Type:** `{content_type}`  
                                """)
                                st.write(chunk.page_content[:500] + "...")
                else:
                    st.error(answer)

if __name__ == "__main__":
    main()
