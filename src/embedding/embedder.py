from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from google.genai import types
from services.gemini import client as gemini_client
from langfuse import observe
import os
from dotenv import load_dotenv
from pydantic import BaseModel, PrivateAttr

import google.genai as genai
from google.genai import types
from google.genai.types import Content, Part
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun, CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
import torch
from sentence_transformers import CrossEncoder
from typing import List, Callable, Any, Tuple, Optional


class VertexAIChat(BaseChatModel, BaseModel):
    model: str
    temperature: float = 0.0
    max_tokens: int = None
    top_p: float = 0.8
    _client: genai.Client = PrivateAttr()

    def __init__(self, client=gemini_client, **kwargs):
        super().__init__(**kwargs)
        self._client = client

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        contents = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "model"
            contents.append(Content(parts=[Part(text=msg.content)], role=role))

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config={"temperature": self.temperature}
        )
        return ChatResult(
            generations=[
                {
                    "text": response.text,
                    "message": AIMessage(content=response.text)
                }
            ]
        )

    def _create_chat_result(self, text: str):
        return self._to_chat_result(AIMessage(content=text))

    def _to_chat_result(self, message: AIMessage):
        from langchain_core.outputs import ChatResult, ChatGeneration
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "vertexai-chat"



class GeminiEmbeddings(Embeddings):
    def __init__(self, client=gemini_client, model: str = "text-embedding-004"):
        self.client = client
        self.model = model

    @observe(as_type="embedding")
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
    
    @observe(as_type="embedding")
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

