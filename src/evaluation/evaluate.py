import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
import pandas as pd
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant

from ragas import SingleTurnSample
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_google_genai import ChatGoogleGenerativeAI

from embedding.embedder import GeminiEmbeddings, VertexAIChat
from retrieval.retriever import retrieve
from services.gemini import call_gemini

load_dotenv()


embedding_model = GeminiEmbeddings()
vectorstore = Qdrant(
    client=QdrantClient(host="localhost", port=6333),
    collection_name="ifc_report",
    embeddings=embedding_model
)

evaluator_llm = LangchainLLMWrapper(VertexAIChat(model="gemini-2.0-flash", temperature=10))

evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)

df = pd.read_csv("./data/RAG_evaluation_dataset.csv")
df = df.head(10)

async def async_call_gemini(context: str, question: str) -> str:
    import asyncio
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, call_gemini, context, question)
    return response

async def main():
    total_faithfulness = 0
    total_context_precision = 0
    total_response_relevance = 0

    for i, row in df.iterrows():

        question = row['Question']
        reference = row['Ground_Truth_Context']

        retrieved_docs = retrieve(vectorstore, question)

        retrieved_contexts = [doc.page_content for doc in retrieved_docs]
        context_snippet = "\n\n".join(retrieved_contexts[:10])

        response_text = await async_call_gemini(context_snippet, question)

        sample_f = SingleTurnSample(
            user_input=question,
            retrieved_contexts=retrieved_contexts,
            response=response_text,
        )
        sample_c = SingleTurnSample(
            user_input=question,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )

        faithfulness_score = await Faithfulness(llm=evaluator_llm).single_turn_ascore(sample_f)
        context_precision_score = await LLMContextPrecisionWithReference(llm=evaluator_llm).single_turn_ascore(sample_c)
        response_relevance_score = await ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings).single_turn_ascore(sample_f)

        total_faithfulness += faithfulness_score
        total_context_precision += context_precision_score
        total_response_relevance += response_relevance_score

        print(f"Faithfulness: {faithfulness_score:.2f}")
        print(f"Context Precision: {context_precision_score:.2f}")
        print(f"Response Relevance: {response_relevance_score:.2f}")

        print(f"Finished sample {i+1}.")


if __name__ == "__main__":
    asyncio.run(main())
