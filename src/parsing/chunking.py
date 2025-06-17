from langchain_experimental.text_splitter import SemanticChunker
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from embedding.embedder import GeminiEmbeddings

def semantic_chunking_with_st(text: str, model_name: str = "all-MiniLM-L6-v2") -> list[Document]:
    embeddings = GeminiEmbeddings()
    chunker = SemanticChunker(embeddings)
    return chunker.create_documents([text])
