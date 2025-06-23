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



client = genai.Client(
    vertexai=os.getenv("USE_VERTEXAI", "False") == "True",
    project=os.getenv("PROJECT_ID"),
    location=os.getenv("LOCATION")
)

@observe(as_type="generation")
def call_gemini(context:str, user_query:str) -> str:
    model = "gemini-2.0-flash"
    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            temperature=0.3,
            system_instruction=(f"""
                You are a helpful assistant answering questions based **only** on the context from the IFC Annual Report 2024. 
                - If the answer is not in the context, say: 'The document does not provide this information.'
                - If the question is not related to the document, say: 'This question is not related to the IFC Annual Report 2024.'
                - If the answer is not in the pages provided, say: 'The answer is not in the provided pages.'"""
            )
        ),
        contents=f"""Answer the question using the following context:\n{context}\n\nQuestion: {user_query}"""
    )

    return response.text


from pydantic import BaseModel

class Metadata(BaseModel):
    query: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    _type: str | None = None


@observe(as_type="generation")
def return_metadata(user_query: str):
    response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=f"You are a helpful assistant that returns metadata based on user queries. User query: {user_query}",
    config={
        "response_mime_type": "application/json",
        "response_schema": Metadata,
    })
    metadata = json.loads(response.text)
    metadata = Metadata(**metadata)
    return metadata.query, metadata.page_start, metadata.page_end, metadata.section, metadata._type


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




