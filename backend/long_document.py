import re
from collections import defaultdict

from langchain_core.documents import Document


class LongDocumentPolicy:
    LARGE_THRESHOLD = 21

    def __init__(self, neighbor_window: int = 4, detail_chunks: int = 48, summary_window_chunks: int = 8):
        self.neighbor_window = max(0, neighbor_window)
        self.detail_chunks = max(1, detail_chunks)
        self.summary_window_chunks = max(1, summary_window_chunks)

    def apply(self, route_name: str, docs: list[Document], query: str):
        grouped = self.group_by_page(docs)
        if not grouped:
            return docs, {"long_document_mode": "none"}

        _, target_docs = max(
            grouped.items(),
            key=lambda item: (self.page_score(item[1], query), len(item[1])),
        )
        chunk_count = len(target_docs)
        if chunk_count < self.LARGE_THRESHOLD:
            return docs, {"long_document_mode": "none", "document_chunk_count": chunk_count}

        if route_name == "long_doc_summary":
            return target_docs, {
                "long_document_mode": "summary",
                "document_chunk_count": chunk_count,
                "selected_chunk_indexes": self.selected_chunk_indexes(target_docs),
                "map_reduce_passes": len(self.summary_windows(target_docs)),
            }
        if route_name == "long_doc_detail":
            selected = self.select_detail_window(target_docs)
            return selected, self.stats("detail", chunk_count, selected)

        selected, stats = self.select_targeted_window(target_docs, query)
        return selected, stats

    def group_by_page(self, docs: list[Document]) -> dict[str, list[Document]]:
        grouped = defaultdict(list)
        for doc in docs:
            page_key = doc.metadata.get("page_id") or doc.metadata.get("url") or doc.metadata.get("title")
            if page_key:
                grouped[str(page_key)].append(doc)
        for page_docs in grouped.values():
            page_docs[:] = self.dedupe_and_sort(page_docs)
        return dict(grouped)

    def select_targeted_window(self, docs: list[Document], query: str):
        docs = self.dedupe_and_sort(docs)
        if not docs:
            return [], self.stats("lookup", 0, [])

        anchor = max(docs, key=lambda doc: (self.chunk_score(doc, query, include_metadata=False), -self.chunk_index(doc)))
        anchor_index = self.chunk_index(anchor)
        start = max(0, anchor_index - self.neighbor_window)
        end = anchor_index + self.neighbor_window
        selected = [doc for doc in docs if start <= self.chunk_index(doc) <= end]
        return selected, self.stats("lookup", len(docs), selected)

    def select_detail_window(self, docs: list[Document]) -> list[Document]:
        return self.dedupe_and_sort(docs)[: self.detail_chunks]

    def summary_windows(self, docs: list[Document]) -> list[list[Document]]:
        ordered = self.dedupe_and_sort(docs)
        size = self.summary_window_chunks
        return [ordered[index : index + size] for index in range(0, len(ordered), size)]

    def page_score(self, docs: list[Document], query: str) -> int:
        return sum(self.chunk_score(doc, query, include_metadata=True) for doc in docs)

    def chunk_score(self, doc: Document, query: str, include_metadata: bool = True) -> int:
        terms = self.query_terms(query)
        if not terms:
            return 0
        values = [doc.page_content]
        if include_metadata:
            values.extend([doc.metadata.get("title", ""), doc.metadata.get("breadcrumb", "")])
        haystack = " ".join(str(value) for value in values).lower()
        return sum(1 for term in terms if term in haystack)

    def query_terms(self, query: str) -> set[str]:
        return {
            term.lower()
            for term in re.findall(r"[0-9A-Za-z가-힣]+", query or "")
            if len(term) > 1
        }

    def chunk_index(self, doc: Document) -> int:
        try:
            return int(doc.metadata.get("chunk_index", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def dedupe_and_sort(self, docs: list[Document]) -> list[Document]:
        deduped = {}
        for doc in docs:
            key = (
                doc.metadata.get("page_id") or doc.metadata.get("url") or doc.metadata.get("title"),
                self.chunk_index(doc),
                doc.metadata.get("_id"),
            )
            simple_key = key[:2]
            if simple_key not in deduped:
                deduped[simple_key] = doc
        return sorted(deduped.values(), key=self.chunk_index)

    def selected_chunk_indexes(self, selected: list[Document]) -> list[int]:
        return [self.chunk_index(doc) for doc in selected]

    def stats(self, mode: str, chunk_count: int, selected: list[Document]) -> dict:
        return {
            "long_document_mode": mode,
            "document_chunk_count": chunk_count,
            "selected_chunk_indexes": self.selected_chunk_indexes(selected),
            "map_reduce_passes": 0,
        }
