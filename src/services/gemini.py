from google import genai
import os
from dotenv import load_dotenv
load_dotenv()
from langfuse import observe
from google.genai import types
#from langfuse.decorators import observe, langfuse_context



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

    # langfuse_context.update_current_observation(
    #     input=input,
    #     model=model,
    #     usage_details={
    #         "input": response.usage_metadata.prompt_token_count,
    #         "output": response.usage_metadata.candidates_token_count,
    #         "total": response.usage_metadata.total_token_count
    #     })
    return response.text



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



