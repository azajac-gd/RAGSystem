from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from google.genai import types
from services.gemini import client as gemini_client
from langfuse import observe



class GeminiEmbeddings(Embeddings):
    def __init__(self, client=gemini_client, model: str = "text-embedding-004"):
        self.client = client
        self.model = model
    @observe()
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        batch_size = 16
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            responses = [
                self.client.models.embed_content(
                    model=self.model,
                    contents={"parts": [{"text": text}]},
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                for text in batch
            ]
            embeddings.extend([r.embeddings[0].values for r in responses])
        return embeddings
    @observe()
    def embed_query(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents={"parts": [{"text": text}]},
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        return response.embeddings[0].values


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

