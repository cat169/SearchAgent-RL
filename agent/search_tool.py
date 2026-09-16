from typing import Optional

from retrieval.bm25_retriever import BM25Retriever


class SearchTool:
    def __init__(
        self,
        retriever: BM25Retriever,
        default_top_k: int = 3,
        max_text_chars: Optional[int] = None,
    ):
        if default_top_k < 1:
            raise ValueError("default_top_k must be >= 1")

        if max_text_chars is not None and max_text_chars < 1:
            raise ValueError("max_text_chars must be >= 1 or None")

        self.retriever = retriever
        self.default_top_k = default_top_k
        self.max_text_chars = max_text_chars

    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> str:
        effective_top_k = (
            self.default_top_k
            if top_k is None
            else top_k
        )

        results = self.retriever.search(
            query=query,
            top_k=effective_top_k,
        )

        return self._format_results(results)

    def batch_run(
        self,
        queries: list[str],
        top_k: Optional[int] = None,
        threads: int = 1,
    ) -> list[str]:
        effective_top_k = (
            self.default_top_k
            if top_k is None
            else top_k
        )

        batch_results = self.retriever.batch_search(
            queries=queries,
            top_k=effective_top_k,
            threads=threads,
        )

        return [
            self._format_results(results)
            for results in batch_results
        ]



    def _format_results(
        self,
        results: list[dict],
    ) -> str:
        formatted_documents = []

        for result in results:
            text = result["text"]

            if (
                self.max_text_chars is not None
                and len(text) > self.max_text_chars
            ):
                text = text[: self.max_text_chars].rstrip()

            document = (
                f"[{result['rank']}] "
                f"Title: {result['title']}\n"
                f"Content: {text}"
            )

            formatted_documents.append(document)

        return "\n\n".join(formatted_documents)