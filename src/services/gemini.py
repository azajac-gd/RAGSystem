from google import genai
import os
from dotenv import load_dotenv
load_dotenv()
from langfuse import observe
from google.genai import types
#from langfuse.decorators import observe, langfuse_context
import json
import io
from PIL import Image
import pandas as pd
from pydantic import BaseModel
from enum import Enum
from langchain.schema import Document
import logging



client = genai.Client(
    vertexai=os.getenv("USE_VERTEXAI", "False") == "True",
    project=os.getenv("PROJECT_ID"),
    location=os.getenv("LOCATION")
)

def build_context_with_documents(docs: list[Document]) -> str:
    context_parts = []
    for doc in docs:
        doc_type = doc.metadata.get("type")
        
        if doc_type == "table":
            csv_path = doc.metadata.get("csv_path")
            if csv_path and os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                csv_preview = df.head(10).to_markdown(index=False)
                context = f"[TABLE] Summary: {doc.page_content}\nTable Preview:\n{csv_preview}"
            else:
                context = f"[TABLE] Summary: {doc.page_content} (table file missing)"
        
        elif doc_type == "image":
            context = f"[IMAGE] Summary: {doc.page_content}"
        
        elif doc_type == "text":
            context = f"[TEXT] {doc.page_content}"
        
        else:
            context = f"[UNKNOWN TYPE] {doc.page_content}"

        context_parts.append(context)

    return "\n\n---\n\n".join(context_parts)


@observe(as_type="generation")
def call_gemini(docs: list[Document], user_query: str) -> str:
    context = build_context_with_documents(docs)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            temperature=0.3,
            system_instruction=(
                "You are a helpful assistant answering questions based **only** on the context from the IFC Annual Report 2024. "
                "- If the answer is not in the context, say: 'The document does not provide this information.' "
                "- If the question is not related to the document, say: 'This question is not related to the IFC Annual Report 2024.' "
                "- If the answer is not in the pages provided, say: 'The answer is not in the provided pages.'"
            ),
        ),
        contents=f"""Answer the question using the following context:\n{context}\n\nQuestion: {user_query}"""
    )

    return response.text



class ContentType(str, Enum):
    text = "text"
    image = "image"
    table = "table"

class Metadata(BaseModel):
    query: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    _type: ContentType | None = None 


# @observe(as_type="generation")
# def return_metadata(user_query: str):
#     response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents=f"""
#     You are a helpful assistant that extracts structured metadata from the user query.
#     Your task is to return the following fields in JSON:

#     - query: restate the original user query.
#     - page_start: the starting page number of the relevant section (if applicable).
#     - page_end: the ending page number of the relevant section (if applicable).
#     - section: name of the report section (if applicable).
#     - _type: classify the content type the user is referring to — choose only from ["text", "image", "table"].

#     When users ask for numeric data, financial breakdowns, rows/columns, specific values, or metrics, you should classify the type as "table".

#     User query: {user_query}
#     """,
#     config={
#         "response_mime_type": "application/json",
#         "response_schema": Metadata,
#     })
#     metadata = json.loads(response.text)
#     metadata = Metadata(**metadata)
#     return metadata.query, metadata.page_start, metadata.page_end, metadata.section, metadata._type


@observe(as_type="generation")
def summarize_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part(text="What does this chart or image show? Summarize it briefly."),
            types.Part(inline_data={"mime_type": "image/png", "data": image_bytes})
        ]
    )
    return response.text


@observe(as_type="generation")
def summarize_table(csv_path):
    df = pd.read_csv(csv_path)
    table_string = df.to_csv(index=False)

    prompt = f"""This is a table extracted from a financial report PDF. Please summarize what this table is about, what kind of data it presents, and if possible identify any patterns.

        CSV Table:
        {table_string}

        Brief summary:"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text.strip()


search_with_metadata_declaration = {
    "name": "search_with_metadata",
    "description": "Return metadata for the search based on the user query.",
    "parameters":{
        "type": "object",
        "properties": {
            "query": {"type": "string", 
                      "description": "A keyword or phrase to search for."},
            "page_start": {"type": "integer", 
                           "description": "Starting page number (optional).", 
                           "nullable": True},
            "page_end": {"type": "integer", 
                         "description": "Ending page number (optional).", 
                         "nullable": True},
            "section": {"type": "string", 
                        "description": "Document section.", 
                        "nullable": True},
             
            "type": {"type": "string", 
                     "description": "Type of content: 'text', 'table', or 'figure'", 
                     "nullable": True},
        },
        "required": ["query"]
    }
}

@observe(as_type="generation")
def return_metadata(user_query: str) -> str:
    model = "gemini-2.0-flash"
    tools = types.Tool(function_declarations=[search_with_metadata_declaration])
    config = types.GenerateContentConfig(
        temperature=0.0,
        system_instruction=f"""You are a helpful assistant that returns metadata for the search based on the user query.""",
        tools=[tools],
        tool_config= {"function_calling_config": {"mode": "any"}})
    
    contents = [
        types.Content(
            role="user", parts=[types.Part(text=f"User query: {user_query}")]
        )
    ]
    response = client.models.generate_content(
        model=model, config=config, contents=contents
    )

    tool_call = response.candidates[0].content.parts[0].function_call
    if tool_call is None:
        return "Model did not choose any tool."

    name = tool_call.name
    args = tool_call.args

    if name == "search_with_metadata":
        query = args.get("query")
        page_start = args.get("page_start")
        if page_start is not None:
            page_start = int(page_start)
        page_end = args.get("page_end")
        if page_end is not None:
            page_end = args.get("page_end")
        section = args.get("section")
        type_ = args.get("type")
        return query, page_start, page_end, section, type_