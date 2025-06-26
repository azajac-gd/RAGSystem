from google import genai
import os
from dotenv import load_dotenv
load_dotenv()
from langfuse import observe
from google.genai import types
import json
import io
from PIL import Image
from pydantic import BaseModel
from langchain.schema import Document
from typing import Optional, Literal


client = genai.Client(
    vertexai=os.getenv("USE_VERTEXAI", "False") == "True",
    project=os.getenv("PROJECT_ID"),
    location=os.getenv("LOCATION")
)


@observe(as_type="generation")
def call_gemini(docs: list[Document], user_query: str) -> str:
    context = "\n\n".join(doc.page_content for doc in docs)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            temperature=0.3,
            system_instruction=(
                "You are a helpful assistant answering questions based **only** on the context from the IFC Annual Report 2024. "
                "- If the answer is not in the context, say: 'The document does not provide this information.' "
                "- If the question is not related to the document, say: 'This question is not related to the IFC Annual Report 2024.' "
            ),
        ),
        contents=f"""Answer the question based strictly on the following context:\n\n{context}\n\nQuestion: {user_query}"""
    )

    return response.text


class Metadata(BaseModel):
    query: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    content_type: Optional[Literal["text", "image", "table"]] = None

@observe(as_type="generation")
def return_metadata(user_query: str):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"""
You are a helpful assistant that extracts structured metadata from a user query.
Your task is to return the following fields in JSON format:

- query: restate the original user query.
- page_start: the starting page number of the relevant section (if applicable).
- page_end: the ending page number of the relevant section (if applicable).
- content_type: ONLY IF the user explicitly asks for a specific content type, return one of ["text", "image", "table"].
  If the user does not request a specific type, return null.

Guidelines:
- If the user asks for descriptions, explanations, summaries → content_type = "text"
- If the user asks for charts, diagrams, figures, visual data → content_type = "image"
- If the user asks for numbers, values, breakdowns, metrics, columns, financial data → content_type = "table"
- If it's ambiguous or general, set content_type to null.

User query: {user_query}
""",
        config={
            "response_mime_type": "application/json",
            "response_schema": Metadata,
        }
    )
    metadata = json.loads(response.text)
    metadata = Metadata(**metadata)
    return metadata.query, metadata.page_start, metadata.page_end, metadata.content_type


@observe(as_type="generation")
def summarize_image(image_bytes, title):
    image = Image.open(io.BytesIO(image_bytes))
    prompt = f"""
    You are an expert analyst specialized in describing **data visualizations and diagrams** (such as charts, plots, and schematics).

    Your task is to generate a highly informative and structured summary of the image, as if preparing it for detailed question answering or extraction. The summary must be precise and descriptive enough to answer any question a user may ask about the image.

    Instructions:
    - If a meaningful title is provided (i.e., not "Unknown"), use it as a starting point.
    - Identify the **type of chart or diagram** (e.g., bar chart, line graph, pie chart, flow diagram, etc.).
    - Describe:
    - The **axes** (labels, units, ranges).
    - The **main variables** or categories shown.
    - Any **trends, correlations, or outliers**.
    - **Colors or legends**, and what they represent.
    - The **time range or data scope** if present.
    - If labels or values are **too small or blurry**, try to **infer** their purpose from context and say so.
    - If it's a schematic or diagram: describe the elements, structure, and how the components relate.

    Do not use generic phrases like "this image shows". Do not repeat instructions.

    Example format:
    **Title**: (use provided or inferred)
    **Type**: (e.g. stacked bar chart)
    **Axes**: X – [label, unit, range]; Y – [label, unit, range]
    **Legend**: ...
    **Summary**: ...

    Hint title: "{title}"
        """.strip()

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part(text=prompt),
            types.Part(inline_data={"mime_type": "image/png", "data": image_bytes}),
        ]
    )
    return response.text
