from __future__ import annotations

from pathlib import Path

from voice_rag_agent.rag.documents import DocumentChunk
from voice_rag_agent.rag.retriever import RetrievedChunk


class LlamaIndexRetrieverAdapter:
    """Adapter for teams that want LlamaIndex-backed retrieval in production."""

    def __init__(
        self,
        data_dir: Path,
        top_k: int = 4,
        ollama_model: str = "gemma3:4b",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        try:
            from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from llama_index.llms.ollama import Ollama
        except ImportError as exc:
            raise RuntimeError("Install the rag extra to use LlamaIndex retrieval.") from exc

        Settings.llm = Ollama(model=ollama_model)
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        documents = SimpleDirectoryReader(str(data_dir)).load_data()
        self._retriever = VectorStoreIndex.from_documents(documents).as_retriever(
            similarity_top_k=top_k
        )

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        del top_k
        nodes = self._retriever.retrieve(query)
        results: list[RetrievedChunk] = []
        for index, node_with_score in enumerate(nodes):
            node = node_with_score.node
            metadata = dict(node.metadata or {})
            source_path = str(metadata.get("file_path") or metadata.get("path") or "llama-index")
            title = str(
                metadata.get("file_name") or metadata.get("title") or Path(source_path).name
            )
            score = float(node_with_score.score or 0.0)
            results.append(
                RetrievedChunk(
                    chunk=DocumentChunk(
                        id=str(getattr(node, "node_id", f"llama_{index}")),
                        source_path=source_path,
                        text=node.get_content(),
                        metadata={
                            "title": title,
                            "path": source_path,
                            "chunk_index": index,
                        },
                    ),
                    score=score,
                )
            )
        return results
