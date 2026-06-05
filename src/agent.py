from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        context_blocks = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source = metadata.get("source") or metadata.get("doc_id") or result.get("doc_id", "unknown")
            metadata_lines = [
                f"Title: {metadata.get('title', '')}",
                f"Source: {source}",
                f"URL: {metadata.get('url', '')}",
                f"Original price: {metadata.get('original_price', '')}",
                f"Current price: {metadata.get('current_price', '')}",
                f"Section: {metadata.get('section', '')}",
                f"Chunk index: {metadata.get('chunk_index', '')}",
                f"Chunk strategy: {metadata.get('chunk_strategy', '')}",
                f"Document ID: {metadata.get('doc_id', result.get('doc_id', ''))}",
            ]
            context_blocks.append(
                f"[Context {index}]\n"
                + "\n".join(metadata_lines)
                + f"\nContent:\n{result['content']}"
            )

        context = "\n\n".join(context_blocks)
        prompt = f"""
            You are a helpful knowledge base assistant.
            Answer the user's question using only the context below.
            Always answer in Vietnamese.
            If the context does not contain enough information, say you do not know.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """.strip()

        return self.llm_fn(prompt)
