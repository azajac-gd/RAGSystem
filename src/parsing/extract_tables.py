import pandas as pd
from langchain.schema import Document
import pdfplumber
import pandas as pd
import json
from services.gemini import summarize_table

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

#extract_tables_with_pdfplumber(pdf_path, output_dir="output_tables")    

def generate_table_summaries(metadata_path="output_tables/metadata.json"):
    with open(metadata_path) as f:
        metadata_list = json.load(f)

    for entry in metadata_list:
        csv_path = entry["csv_path"]
        try:
            summary = summarize_table(csv_path)
            entry["summary"] = summary
            print(f"Summarized table on page {entry['page']}, table {entry['table_number']}")
        except Exception as e:
            print(f"Error summarizing table at {csv_path}: {e}")
            entry["summary"] = None

    with open(metadata_path, "w") as f:
        json.dump(metadata_list, f, indent=2)

    return metadata_list

#generate_table_summaries("output_tables/metadata.json")
