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

                        }
            documents.append(Document(page_content=cleaned, metadata=metadata))
    
    return documents


def extract_text_with_docling(pdf_path):
    converter = DocumentConverter()
    pdf = converter.convert(source=pdf_path).document

    text_blocks = [block.text for block in pdf.text_blocks if block.text.strip()]
    full_text = "\n\n".join(text_blocks)

    return full_text
