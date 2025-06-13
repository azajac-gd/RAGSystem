import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embedding.embedder import SentenceTransformerEmbeddings
from embedding.index_faiss import build_qdrant_index_with_sentence_transformer
from parsing.extract_text import extract_text
from retrieval.retriever import retrieve
from llm.gemini import call_gemini
from parsing.chunking import semantic_chunking_with_st
import logging


PDF_PATH = "./data/pdf/ifc-annual-report-2024-financials.pdf"
QDRANT_COLLECTION = "ifc_report"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@st.cache_resource
def load_and_index_documents():
    docs = extract_text(PDF_PATH)
    full_text = " ".join([doc.page_content for doc in docs])
    documents = semantic_chunking_with_st(full_text)
    for i in range(10):
        st.info(documents[i].page_content)  
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = build_qdrant_index_with_sentence_transformer(
        documents,
        embedding_model,
        collection_name=QDRANT_COLLECTION
    )
    return vectorstore


# === Streamlit App ===
def main():
    st.title("IFC Annual Report RAG System")
    user_query = st.text_input("Ask a question about the report")

    with st.spinner("Loading..."):
        vectorstore = load_and_index_documents()

    if user_query:
        with st.spinner("Processing query..."):
            relevant_chunks = retrieve(user_query, vectorstore)
            context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
            prompt = f"Answer the question using the following context:\n{context}\n\nQuestion: {user_query}"
            answer = call_gemini(prompt)

            st.markdown("### Answer:")
            st.write(answer)

            st.markdown("### Retrieved Context:")
            for chunk in relevant_chunks:
                st.markdown(f"> {chunk.page_content}")


if __name__ == "__main__":
    main()
