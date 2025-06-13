from google import genai
import os
from dotenv import load_dotenv
load_dotenv()
from langfuse import observe


client = genai.Client(
    vertexai=os.getenv("USE_VERTEXAI", "False") == "True",
    project=os.getenv("PROJECT_ID"),
    location=os.getenv("LOCATION")
)

@observe(as_type="generation")
def call_gemini(prompt):
    model = "gemini-2.0-flash"
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    return response.text

