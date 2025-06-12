from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document

def clean_text(text):
    return text.strip()

def extract_text(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    return [
        Document(page_content=clean_text(doc.page_content), metadata=doc.metadata)
        for doc in docs
    ]