import os
from pathlib import Path
from typing import Optional

from datasets import load_dataset


class BM25Retriever:
    def __init__(
        self,
        index_path: str,
        corpus_path: str,
        top_k: int = 3,
        cache_dir: Optional[str] = None,
    ):
        self.index_path = Path(index_path)
        self.corpus_path = Path(corpus_path)
        self.top_k = top_k

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"BM25 index path does not exist: {self.index_path}"
            )

        if not self.corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus path does not exist: {self.corpus_path}"
            )

        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")

        if cache_dir is None:
            self.cache_dir = self.corpus_path.parent / ".hf_cache"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Pyserini 0.44.0 import-time compatibility workaround.
        # Sparse Lucene BM25 itself does not use the OpenAI API.
        os.environ.setdefault(
            "OPENAI_API_KEY",
            "local-placeholder-pyserini-import-only",
        )

        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(str(self.index_path))

        self.corpus = load_dataset(
            "json",
            data_files=str(self.corpus_path),
            split="train",
            cache_dir=str(self.cache_dir),
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        query = query.strip()

        if not query:
            raise ValueError("query must not be empty")

        effective_top_k = self.top_k if top_k is None else top_k

        if effective_top_k < 1:
            raise ValueError("top_k must be >= 1")

        hits = self.searcher.search(query, effective_top_k)

        results = []

        for rank, hit in enumerate(hits, start=1):
            doc = self.corpus[int(hit.docid)]
            contents = doc["contents"]

            if "\n" in contents:
                title, text = contents.split("\n", 1)
            else:
                title = ""
                text = contents

            results.append(
                {
                    "rank": rank,
                    "doc_id": str(hit.docid),
                    "title": title.strip().strip('"'),
                    "text": text.strip(),
                    "score": float(hit.score),
                }
            )

        return results

    def batch_search(
        self,
        queries: list[str],
        top_k: Optional[int] = None,
        threads: int = 1,
    ) -> list[list[dict]]:
        if not queries:
            return []

        cleaned_queries = []

        for query in queries:
            query = query.strip()

            if not query:
                raise ValueError(
                    "queries must not contain empty queries"
                )

            cleaned_queries.append(query)

        effective_top_k = (
            self.top_k
            if top_k is None
            else top_k
        )

        if effective_top_k < 1:
            raise ValueError("top_k must be >= 1")

        if threads < 1:
            raise ValueError("threads must be >= 1")

        qids = [
            str(index)
            for index in range(len(cleaned_queries))
        ]

        batch_hits = self.searcher.batch_search(
            queries=cleaned_queries,
            qids=qids,
            k=effective_top_k,
            threads=threads,
        )

        batch_results = []

        for qid in qids:
            hits = batch_hits[qid]

            results = []

            for rank, hit in enumerate(
                hits,
                start=1,
            ):
                doc = self.corpus[int(hit.docid)]
                contents = doc["contents"]

                if "\n" in contents:
                    title, text = contents.split(
                        "\n",
                        1,
                    )
                else:
                    title = ""
                    text = contents

                results.append(
                    {
                        "rank": rank,
                        "doc_id": str(hit.docid),
                        "title": title.strip().strip('"'),
                        "text": text.strip(),
                        "score": float(hit.score),
                    }
                )

            batch_results.append(results)

        return batch_results