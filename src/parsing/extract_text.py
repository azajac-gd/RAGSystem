import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.schema import Document
from docling.document_converter import DocumentConverter
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
import re

def clean_text(text: str) -> str:
    #Replace line breaks
    text = text.replace('\n', ' ').replace('\r', ' ')
    #Remove page numbers
    text = re.sub(r'\bIFC 2024 ANNUAL REPORT FINANCIALS\s*\d+\b', '', text, flags=re.IGNORECASE)
    #Remove multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    #Remove horizontal lines
    text = re.sub(r'[─━═‾_—–-]{3,}', '', text)
    #Remove figure and table references
    text = re.sub(r'(Figure|Table)\s?\d+[:.]?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\.{4,}', '', text)

    return text.strip()


def extract_text(pdf_path):
    documents = []

    for page_number, page_layout in enumerate(extract_pages(pdf_path), start=0):
        texts = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                texts.append(element.get_text())

        if texts:
            page_text = " ".join(texts)
            cleaned = clean_text(page_text)
            metadata = {"page": page_number,
                        "section": "IFC 2024 Annual Report Financials",
                        "type": "text"}
                        
            documents.append(Document(page_content=cleaned, metadata=metadata))
    
    return documents


def extract_text_with_docling(pdf_path):
    converter = DocumentConverter()
    pdf = converter.convert(source=pdf_path).document

    cleaned_docs = []
    for doc in pdf:
        cleaned_content = clean_text(doc.page_content)
        cleaned_docs.append(Document(page_content=cleaned_content, metadata=doc.metadata))

    return cleaned_docs

