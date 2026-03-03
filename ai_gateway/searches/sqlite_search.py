import json
import os.path
import re
import sqlite3
from typing import Any, List, override

import structlog

from ai_gateway.searches.typing import SearchResult

from .search import Searcher

log = structlog.stdlib.get_logger("chat")


class SqliteSearch(Searcher):

    def __init__(self, *_args, **_kwargs):
        self.db_path = os.path.join("tmp", "docs.db")

    @override
    async def search(
        self,
        query: str,
        gl_version: str,
        page_size: int = 20,
        **kwargs: Any,
    ) -> List[SearchResult]:
        if os.path.isfile(self.db_path):
            conn = sqlite3.connect(self.db_path)
            indexer = conn.cursor()
        else:
            conn = None
            indexer = None

        if not indexer:
            log.warning("SqliteSearch: No database found for documentation searches.")

            return []

        # We need to remove punctuation because table was created with FTS5
        # see https://stackoverflow.com/questions/46525854/sqlite3-fts5-error-when-using-punctuation
        sanitized_query = re.sub(r"[^\w\s]", "", query, flags=re.UNICODE)

        data = indexer.execute(
            "SELECT metadata, content FROM doc_index WHERE processed MATCH ? ORDER BY bm25(doc_index) LIMIT ?",
            (sanitized_query, page_size),
        )

        parsed_response: list[dict] = self._parse_response(data)

        if conn:
            conn.close()

        results, token_count = self.limit_search_results(
            parsed_response, max_tokens=8000
        )

        self.log_search_results(query, page_size, gl_version, results, token_count)

        return results

    @override
    def provider(self):
        return "sqlite"

    def _parse_response(self, response) -> list[dict]:
        results = []

        for r in response:
            metadata = json.loads(r[0])
            search_result = {
                "id": metadata["filename"],
                "content": r[1],
                "metadata": metadata,
            }
            results.append(search_result)
        return results

    def limit_search_results(
        self, response: list[dict], max_tokens: int
    ) -> tuple[list[SearchResult], int]:
        """Limit search results based on a maximum token count."""
        token_count = 0
        results = []

        for result in response:
            tokens = self.estimate_token_count(result["content"])
            if token_count + tokens > max_tokens:
                break
            token_count += tokens
            results.append(
                {
                    "id": result["id"],
                    "content": result["content"],
                    "metadata": result["metadata"],
                }
            )
        search_results = [
            SearchResult(
                id=res["id"],
                content=res["content"],
                metadata=res["metadata"],
            )
            for res in results
        ]

        return search_results, token_count

    def estimate_token_count(self, text: str) -> int:
        """Estimate the number of tokens in a given text (approx: 1.4 x word count)."""
        return int(len(text.split()) * 1.4)
