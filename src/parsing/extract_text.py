from PyPDF2 import PdfReader
from langchain.schema import Document

import re

def clean_text(text: str) -> str:
    #Replace line breaks
    text = text.replace('\n', ' ').replace('\r', ' ')
    #Remove page numbers like "Page 12"
    text = re.sub(r'\bPage\s*\d+\b', '', text, flags=re.IGNORECASE)
    #Remove multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    #Remove horizontal lines or visual dividers (e.g., ─── or ===)
    text = re.sub(r'[─━═‾_—–-]{3,}', '', text)
    #Remove repeated headers or footers
    text = re.sub(r'IFC\s+2024\s+Annual\s+Report', '', text, flags=re.IGNORECASE)
    #Remove figure and table references ("Figure 5:", "Table 2.")
    text = re.sub(r'(Figure|Table)\s?\d+[:.]?', '', text, flags=re.IGNORECASE)
    return text.strip()

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    documents = []
    
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            cleaned = clean_text(text)
            metadata = {
                "page": page_number + 1, 
            }
            documents.append(Document(page_content=cleaned, metadata=metadata))
    
    return documents
