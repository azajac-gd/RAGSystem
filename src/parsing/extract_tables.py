import pandas as pd
from langchain.schema import Document
import pdfplumber
import pandas as pd
import json

pdf_path = "./data/pdf/ifc-annual-report-2024-financials.pdf"

def extract_tables_with_pdfplumber(pdf_path, output_dir="output_tables"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    all_metadata = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()

            for j, table in enumerate(tables):
                if not table:
                    continue

                df = pd.DataFrame(table)

                csv_path = f"{output_dir}/table_page_{i+1}_{j+1}.csv"
                df.to_csv(csv_path, index=False)

                metadata = {
                    "page": i + 1,
                    "table_number": j + 1,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "csv_path": csv_path
                }
                all_metadata.append(metadata)

    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2)

    return all_metadata

extract_tables_with_pdfplumber(pdf_path, output_dir="output_tables")    

def load_tables_from_csvs(csv_dir):
    import os
    documents = []

    for fname in os.listdir(csv_dir):
        if fname.endswith(".csv"):
            path = os.path.join(csv_dir, fname)
            df = pd.read_csv(path)

            table_text = df.to_markdown(index=False)

            metadata = {
                "source": fname,
                "type": "table"
            }

            doc = Document(page_content=table_text, metadata=metadata)
            documents.append(doc)

    return documents
