from google import genai
import os
from dotenv import load_dotenv
load_dotenv()
from langfuse import observe
from google.genai import types


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
                If the answer is not in the context, say: 'The document does not provide this information.'  
                """
            )
        ),
        contents=f"""Answer the question using the following context:\n{context}\n\nQuestion: {user_query}"""
    )

    return response.text

