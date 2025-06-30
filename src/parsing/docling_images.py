import logging
import time
import re
import json
import csv
from pathlib import Path

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem, TextItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain.schema import Document

_log = logging.getLogger(__name__)

pdf_path = "./data/pdf/ifc-annual-report-2024-financials.pdf"
shorter = "./data/pdf/shorter.pdf"
IMAGE_RESOLUTION_SCALE = 2.0



def main():
    logging.basicConfig(level=logging.INFO)

    input_doc_path = pdf_path
    output_dir = Path("scratch")
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start_time = time.time()
    conv_res = doc_converter.convert(input_doc_path)
    doc_filename = conv_res.input.file.stem

    FIGURE_TITLE_RE = re.compile(r"^Figure\s+\d+[:.]?", re.IGNORECASE)
    TABLE_1_TITLE_RE = re.compile(r"^CONSOLIDATED\b.*")
    TABLE_2_TITLE_RE = re.compile(r"SUPPLEMENTAL\b.*")
    TABLE_TITLE_RE = re.compile(r"^Table\s+\d+[:.]?", re.IGNORECASE)
    SECTION_TITLE_RE = re.compile(r"SECTION\s+[A-Z0-9]+\.", re.IGNORECASE)

    picture_counter = 0
    table_counter = 0
    image_metadata = []
    table_metadata = []
    text_metadata = []

    current_section_title = None
    current_figure_title = None
    current_table_title = None
    content = "MANAGEMENT’S DISCUSSION AND ANALYSIS"
    documents = []


    for element, _ in conv_res.document.iterate_items():
        prov = element.prov
        page_number = prov[0].page_no if prov else None


        if isinstance(element, TextItem):
            text = element.text.strip()
            if FIGURE_TITLE_RE.match(text):
                current_figure_title = text
            if TABLE_TITLE_RE.match(text) or TABLE_1_TITLE_RE.match(text) or TABLE_2_TITLE_RE.match(text):
                current_table_title = text
            if SECTION_TITLE_RE.match(text):
                current_section_title = text
            else:
                text_metadata = {
                    "type": "text",
                    "section": current_section_title or "Unknown",
                    "content": content or "Unknown",
                    "page": page_number or -1
                }
                documents.append(Document(page_content=text, metadata=text_metadata))

        if page_number <= 4:
            current_section_title = None
        if page_number >=57:
            current_section_title = None
        if page_number >= 71:
            current_table_title = None
            content = "CONSOLIDATED FINANCIAL STATEMENTS AND INTERNAL CONTROL REPORTS"
        if page_number >= 141:
            content = "INVESTMENT PORTFOLIO"

    #     if isinstance(element, PictureItem):
    #         picture_counter += 1
    #         fn = output_dir / f"{doc_filename}-picture-{picture_counter}.png"
    #         with fn.open("wb") as fp:
    #             element.get_image(conv_res.document).save(fp, "PNG")
    #         image_metadata.append({
    #             "type": "image",
    #             "image_path": str(fn),
    #             "title": current_figure_title or "Unknown",
    #             "page_number": page_number or -1,
    #             "section_title": current_section_title or "Unknown",
    #             "content": content or "Unknown"
    #         })
    #         current_figure_title = None

    #     elif isinstance(element, TableItem):
    #         table_counter += 1
    #         csv_path = output_dir / f"{doc_filename}-table-{table_counter}.csv"

    #         try:
    #             df = element.export_to_dataframe()
    #             df.to_csv(csv_path, index=False)
    #         except Exception as e:
    #             _log.warning(f"Failed to export table {table_counter} on page {page_number}: {e}")

    #         table_metadata.append({
    #             "type": "table",
    #             "csv": str(csv_path),
    #             "title": current_table_title or "Unknown",
    #             "page_number": page_number or -1,
    #             "section_title": current_section_title or "Unknown",
    #             "content": content or "Unknown"
    #         })



    # with open(output_dir / f"{doc_filename}-figures.json", "w") as f:
    #     json.dump(image_metadata, f, indent=2)

    # with open(output_dir / f"{doc_filename}-tables.json", "w") as f:
    #     json.dump(table_metadata, f, indent=2)

    _log.info(f"Document converted. Exported {picture_counter} images and {table_counter} tables in {time.time() - start_time:.2f} seconds.")

    return documents

#main()
