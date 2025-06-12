from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def semantic_chunking_with_st(text: str, model_name: str = "all-MiniLM-L6-v2") -> list[Document]:
    """
    Perform semantic chunking using LangChain's HuggingFaceEmbeddings wrapper.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    chunker = SemanticChunker(embeddings)
    return chunker.create_documents([text])
