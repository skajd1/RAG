import argparse
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from conversation_query import StructuredFollowupQueryResolver
from evaluation.metrics import aggregate_summary, score_case
from evaluation.report import load_golden_cases, render_html_report


def _page(page):
    if isinstance(page, str):
        return {"page_id": page, "title": page}
    return {
        "page_id": str(page.get("page_id") or ""),
        "title": str(page.get("title") or page.get("page_id") or ""),
    }


class FixtureRetrievalAdapter:
    def retrieve(self, case, resolved_query):
        started = perf_counter()
        ranked_pages = case.get("fixture_ranked_pages")
        if ranked_pages is None:
            raise ValueError(f"fixture case {case.get('id', '<unknown>')} requires fixture_ranked_pages")
        return {
            "ranked_pages": [_page(page) for page in ranked_pages],
            "latency_ms": (perf_counter() - started) * 1000,
        }


class LiveRetrievalAdapter:
    def __init__(self, chain_factory=None):
        self.chain_factory = chain_factory

    def retrieve(self, case, resolved_query):
        started = perf_counter()
        try:
            chain = self.chain_factory() if self.chain_factory else self._create_chain()
            if hasattr(chain, "retrieve_context"):
                context = chain.retrieve_context(
                    case["query"],
                    mentions=case.get("mentions", []),
                    history=case.get("history", []),
                    prior_sources=case.get("prior_sources", []),
                )
                docs = context.docs
                route = getattr(context, "route", None)
                long_document_stats = getattr(context, "long_document_stats", {})
            else:
                docs = chain.hybrid_retriever.retrieve(resolved_query).docs
                route = None
                long_document_stats = {}
            ranked_pages = []
            seen_page_ids = set()
            seen_titles = set()
            for doc in docs:
                page = _page(doc.metadata)
                page_id = page["page_id"]
                title = page["title"]
                if not (page_id or title) or page_id in seen_page_ids or title in seen_titles:
                    continue
                if page_id:
                    seen_page_ids.add(page_id)
                if title:
                    seen_titles.add(title)
                ranked_pages.append(page)
        except Exception as error:
            raise RuntimeError(f"live retrieval failed: {error}") from error
        return {
            "ranked_pages": ranked_pages,
            "latency_ms": (perf_counter() - started) * 1000,
            "route": getattr(route, "name", None),
            "route_reason": getattr(route, "reason", None),
            "long_document_stats": long_document_stats,
        }

    def _create_chain(self):
        from rag_chain import RAGChain

        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env")
        load_dotenv(project_root / ".env.local", override=True)
        return RAGChain()


def run_evaluation(
    cases_path,
    output_path,
    mode="fixture",
    k=5,
    adapter=None,
    targets=None,
    baseline_summary=None,
    baseline_case_results=None,
):
    cases = load_golden_cases(cases_path)
    resolver = StructuredFollowupQueryResolver()
    adapter = adapter or (FixtureRetrievalAdapter() if mode == "fixture" else LiveRetrievalAdapter())
    case_results = []
    computed_baseline_results = []

    for case in cases:
        resolved = resolver.resolve(case["query"], case.get("history", []))
        baseline_pages = case.get("baseline_ranked_pages")
        if baseline_pages is not None:
            baseline_retrieved = [
                page["page_id"] or page["title"]
                for page in [_page(page) for page in baseline_pages]
            ]
            computed_baseline_results.append(
                {
                    **case,
                    **score_case(
                        retrieved_pages=baseline_retrieved,
                        relevant_pages=case.get("relevant_pages"),
                        relevance_grades=case.get("relevance_grades"),
                        forbidden_pages=case.get("forbidden_pages"),
                        k=k,
                    ),
                    "retrieved_pages": baseline_retrieved,
                }
            )
        retrieval = adapter.retrieve(case, resolved.retrieval_query)
        ranked_pages = retrieval["ranked_pages"]
        retrieved_pages = [page["page_id"] or page["title"] for page in ranked_pages]
        score = score_case(
            retrieved_pages=retrieved_pages,
            relevant_pages=case.get("relevant_pages"),
            relevance_grades=case.get("relevance_grades"),
            forbidden_pages=case.get("forbidden_pages"),
            k=k,
        )
        case_results.append(
            {
                **case,
                **score,
                "mode": mode,
                "resolved_query": resolved.retrieval_query,
                "scope_changed": resolved.scope_changed,
                "latency_ms": retrieval["latency_ms"],
                "route": retrieval.get("route"),
                "route_reason": retrieval.get("route_reason"),
                "long_document_stats": retrieval.get("long_document_stats", {}),
                "expected_query_match": resolved.retrieval_query == case.get("expected_query"),
                "ranked_pages": ranked_pages,
                "retrieved_pages": retrieved_pages,
            }
        )

    summary = aggregate_summary(case_results)
    if baseline_summary is None and computed_baseline_results:
        baseline_summary = aggregate_summary(computed_baseline_results)
    if baseline_case_results is None and computed_baseline_results:
        baseline_case_results = computed_baseline_results
    category_summaries = {
        category: aggregate_summary(
            result for result in case_results if result.get("category") == category
        )
        for category in sorted({result.get("category", "") for result in case_results})
    }
    html = render_html_report(
        summary,
        category_summaries,
        case_results,
        title=f"RAG Retrieval Quality ({mode})",
        targets=targets,
        baseline_summary=baseline_summary,
        baseline_case_results=baseline_case_results,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return {
        "summary": summary,
        "category_summaries": category_summaries,
        "case_results": case_results,
        "baseline_summary": baseline_summary or {},
        "baseline_case_results": baseline_case_results or [],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run retrieval quality evaluation")
    parser.add_argument("--mode", choices=("fixture", "live"), default="fixture")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run_evaluation(args.cases, args.output, mode=args.mode, k=args.k)


if __name__ == "__main__":
    main()
