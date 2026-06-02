import argparse
import asyncio
import json
import math
import time
from html import escape
from pathlib import Path

from dotenv import load_dotenv

from rag_chain import RAGChain


def load_cases(path):
    cases = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                case = json.loads(line)
                case.setdefault("history", [])
                case.setdefault("expected_facts", [])
                case.setdefault("relevant_pages", [])
                cases.append(case)
    return cases


def normalize(value):
    return "".join(ch for ch in (value or "").lower() if ch.isalnum() or "\uac00" <= ch <= "\ud7a3")


def answer_fact_score(answer, expected_facts):
    if not expected_facts:
        return 1.0, []
    normalized_answer = normalize(answer)
    missing = [fact for fact in expected_facts if normalize(str(fact)) not in normalized_answer]
    return (len(expected_facts) - len(missing)) / len(expected_facts), missing


def source_hit_score(sources, relevant_pages):
    if not relevant_pages:
        return 1.0
    source_keys = {
        str(source.get("page_id") or source.get("title") or "")
        for source in sources
    }
    source_keys.update(str(source.get("title") or "") for source in sources)
    hit_count = sum(1 for page in relevant_pages if str(page) in source_keys)
    return hit_count / len(relevant_pages)


def source_payloads_from_docs(docs):
    return [
        {
            "title": doc.metadata.get("title"),
            "page_id": doc.metadata.get("page_id"),
            "url": doc.metadata.get("url"),
        }
        for doc in docs
    ]


def estimate_tokens(text):
    if not text:
        return 0
    return math.ceil(len(text) / 3)


def selected_context_stats(docs, context_text):
    return {
        "prompt_chunk_count": len(docs),
        "prompt_context_chars": len(context_text),
        "prompt_context_est_tokens": estimate_tokens(context_text),
        "prompt_unique_pages": len(
            {
                doc.metadata.get("page_id") or doc.metadata.get("title") or doc.metadata.get("url")
                for doc in docs
                if doc.metadata.get("page_id") or doc.metadata.get("title") or doc.metadata.get("url")
            }
        ),
    }


def route_context_stats(context_result):
    if not context_result:
        return {}
    route = getattr(context_result, "route", None)
    stats = {
        "route": getattr(route, "name", None),
        "route_reason": getattr(route, "reason", None),
    }
    stats.update(getattr(context_result, "long_document_stats", {}) or {})
    return stats


async def run_case(chain, case):
    started = time.perf_counter()
    answer_parts = []
    sources = []
    finish_reason = None
    async for event in chain.stream_ask(
        case["query"],
        mentions=case.get("mentions", []),
        history=case.get("history", []),
        prior_sources=case.get("prior_sources", []),
    ):
        payload = json.loads(event.strip())
        if payload.get("type") == "answer":
            answer_parts.append(payload.get("content", ""))
        elif payload.get("type") == "sources":
            sources = payload.get("content", [])
        elif payload.get("type") == "done":
            finish_reason = payload.get("finish_reason")

    answer = "".join(answer_parts)
    fact_score, missing_facts = answer_fact_score(answer, case.get("expected_facts", []))
    scored_sources = getattr(chain, "_last_prompt_sources", None) or sources
    context_stats = getattr(chain, "_last_context_stats", {})
    source_score = source_hit_score(scored_sources, case.get("relevant_pages", []))
    return {
        "id": case.get("id", ""),
        "category": case.get("category", ""),
        "query": case["query"],
        "answer": answer,
        "answer_chars": len(answer),
        "answer_est_tokens": estimate_tokens(answer),
        "fact_score": fact_score,
        "source_score": source_score,
        "passed": fact_score >= 0.8,
        "missing_facts": missing_facts,
        "sources": sources,
        "finish_reason": finish_reason,
        "latency_seconds": time.perf_counter() - started,
        **context_stats,
    }


def instrument_prompt_context(chain, chunk_count=None):
    original_format_docs = chain.format_docs

    def format_instrumented_docs(docs):
        selected_docs = docs[:chunk_count] if chunk_count is not None else docs
        chain._last_prompt_sources = source_payloads_from_docs(selected_docs)
        context_text = original_format_docs(selected_docs)
        chain._last_context_stats = selected_context_stats(selected_docs, context_text)
        chain._last_context_stats.update(route_context_stats(getattr(chain, "_last_retrieval_context", None)))
        return context_text

    chain.format_docs = format_instrumented_docs


async def evaluate(cases, context_sizes=None, chunk_counts=None):
    results = []
    if chunk_counts:
        for chunk_count in chunk_counts:
            chain = RAGChain()
            chain.max_context_chars = max(chain.max_context_chars, 50000)
            instrument_prompt_context(chain, chunk_count)
            size_results = []
            for case in cases:
                chain._last_prompt_sources = None
                chain._last_context_stats = {}
                result = await run_case(chain, case)
                result["axis_value"] = chunk_count
                result["axis_label"] = "Context chunks"
                result["context_size"] = chunk_count
                result["chunk_count"] = chunk_count
                size_results.append(result)
            results.extend(size_results)
        return results

    for context_size in context_sizes or []:
        chain = RAGChain()
        chain.max_context_chars = context_size
        instrument_prompt_context(chain)
        size_results = []
        for case in cases:
            chain._last_prompt_sources = None
            chain._last_context_stats = {}
            result = await run_case(chain, case)
            result["axis_value"] = context_size
            result["axis_label"] = "Context chars"
            result["context_size"] = context_size
            size_results.append(result)
        results.extend(size_results)
    return results


def summarize(results):
    summaries = []
    axis_label = results[0].get("axis_label", "Context chars") if results else "Context chars"
    for axis_value in sorted({result["axis_value"] for result in results}):
        group = [result for result in results if result["axis_value"] == axis_value]
        summaries.append(
            {
                "axis_value": axis_value,
                "axis_label": axis_label,
                "context_size": axis_value,
                "case_count": len(group),
                "answer_accuracy": sum(result["fact_score"] for result in group) / len(group),
                "source_accuracy": sum(result["source_score"] for result in group) / len(group),
                "pass_rate": sum(1 for result in group if result["passed"]) / len(group),
                "average_latency_seconds": sum(result["latency_seconds"] for result in group) / len(group),
                "average_prompt_chunks": sum(result.get("prompt_chunk_count", 0) for result in group) / len(group),
                "average_prompt_chars": sum(result.get("prompt_context_chars", 0) for result in group) / len(group),
                "average_prompt_est_tokens": sum(result.get("prompt_context_est_tokens", 0) for result in group) / len(group),
                "average_answer_chars": sum(result.get("answer_chars", 0) for result in group) / len(group),
                "average_answer_est_tokens": sum(result.get("answer_est_tokens", 0) for result in group) / len(group),
            }
        )
    return summaries


def row_axis_value(row):
    return row.get("axis_value", row.get("context_size"))


def bar_chart(title, rows, metric):
    width = 760
    height = 48 + len(rows) * 34
    max_value = max([float(row[metric]) for row in rows] + [1.0])
    parts = []
    for index, row in enumerate(rows):
        y = 36 + index * 34
        value = float(row[metric])
        bar_width = int((width - 240) * min(value / max_value, 1.0))
        parts.append(
            f'<text x="0" y="{y + 15}" class="label">{escape(str(row_axis_value(row)))}</text>'
            f'<rect x="160" y="{y}" width="{bar_width}" height="20" rx="4"></rect>'
            f'<text x="{170 + bar_width}" y="{y + 15}" class="value">{value:.3f}</text>'
        )
    return (
        f"<section><h2>{escape(title)}</h2>"
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        + "".join(parts)
        + "</svg></section>"
    )


def line_chart(rows, metric, title, axis_label=None, value_suffix="", precision=2, y_label=None):
    axis_label = axis_label or (rows[0].get("axis_label") if rows else "Context chars")
    y_label = y_label or title
    width = 860
    height = 360
    plot_left = 78
    plot_top = 34
    plot_width = 710
    plot_height = 240
    if not rows:
        return f"<section><h2>{title}</h2><p>No rows.</p></section>"

    sorted_rows = sorted(rows, key=lambda row: int(row_axis_value(row)))
    max_value = max(float(row.get(metric, 0)) for row in sorted_rows)
    max_value = max(max_value, 1.0)
    x_step = plot_width / max(len(sorted_rows) - 1, 1)

    points = []
    point_labels = []
    x_ticks = []
    for index, row in enumerate(sorted_rows):
        x = plot_left + index * x_step
        value = float(row.get(metric, 0))
        y = plot_top + plot_height - (value / max_value) * plot_height
        points.append((x, y))
        point_labels.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5"></circle>'
            f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" class="value">{value:.{precision}f}{escape(value_suffix)}</text>'
        )
        x_ticks.append(
            f'<line x1="{x:.1f}" y1="{plot_top + plot_height}" x2="{x:.1f}" y2="{plot_top + plot_height + 6}" class="axis"></line>'
            f'<text x="{x:.1f}" y="{plot_top + plot_height + 24}" text-anchor="middle" class="label">{escape(str(row_axis_value(row)))}</text>'
        )

    path = " ".join(("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}" for index, (x, y) in enumerate(points))
    y_ticks = []
    for tick in range(5):
        value = max_value * tick / 4
        y = plot_top + plot_height - (value / max_value) * plot_height
        y_ticks.append(
            f'<line x1="{plot_left - 6}" y1="{y:.1f}" x2="{plot_left}" y2="{y:.1f}" class="axis"></line>'
            f'<text x="{plot_left - 10}" y="{y + 4:.1f}" text-anchor="end" class="label">{value:.1f}{escape(value_suffix)}</text>'
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_left + plot_width}" y2="{y:.1f}" class="grid"></line>'
        )

    return (
        f"<section><h2>{title}</h2>"
        f"<p>X-axis is {escape(axis_label)}. Y-axis is {escape(y_label)}.</p>"
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">'
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}" class="axis"></line>'
        f'<line x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" class="axis"></line>'
        + "".join(y_ticks)
        + "".join(x_ticks)
        + f'<path d="{path}" fill="none" class="line"></path>'
        + "".join(point_labels)
        + f'<text x="{plot_left + plot_width / 2}" y="{height - 14}" text-anchor="middle" class="label">{escape(axis_label)}</text>'
        + f'<text x="18" y="{plot_top + plot_height / 2}" transform="rotate(-90 18 {plot_top + plot_height / 2})" text-anchor="middle" class="label">{escape(y_label)}</text>'
        + "</svg></section>"
    )


def latency_line_chart(rows, axis_label=None):
    return line_chart(
        rows,
        "average_latency_seconds",
        f"{axis_label or 'Context'} vs Average Latency",
        axis_label=axis_label,
        value_suffix="s",
        precision=2,
        y_label="Average latency (seconds)",
    )


def pearson_correlation(rows, x_metric, y_metric):
    pairs = [(float(row.get(x_metric, 0)), float(row.get(y_metric, 0))) for row in rows]
    pairs = [(x, y) for x, y in pairs if x or y]
    if len(pairs) < 2:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denominator_x == 0 or denominator_y == 0:
        return None
    return numerator / (denominator_x * denominator_y)


def render_correlation_table(summaries):
    rows = [
        ("Chunks -> prompt chars", "axis_value", "average_prompt_chars"),
        ("Chunks -> latency", "axis_value", "average_latency_seconds"),
        ("Prompt chars -> latency", "average_prompt_chars", "average_latency_seconds"),
        ("Prompt chars -> answer accuracy", "average_prompt_chars", "answer_accuracy"),
        ("Prompt chars -> source accuracy", "average_prompt_chars", "source_accuracy"),
    ]
    body = ""
    for label, x_metric, y_metric in rows:
        value = pearson_correlation(summaries, x_metric, y_metric)
        display = "n/a" if value is None else f"{value:.3f}"
        body += f"<tr><td>{escape(label)}</td><td>{display}</td></tr>"
    return (
        "<section><h2>Correlation Summary</h2>"
        "<p>Pearson correlation over the summary rows. Values near 1 mean the two metrics rise together.</p>"
        f"<table><thead><tr><th>Relationship</th><th>r</th></tr></thead><tbody>{body}</tbody></table>"
        "</section>"
    )


def render_report(summaries, results, output):
    axis_label = summaries[0].get("axis_label", "Context chars") if summaries else "Context chars"
    summary_rows = "".join(
        "<tr>"
        f"<td>{row_axis_value(row)}</td>"
        f"<td>{row['answer_accuracy']:.3f}</td>"
        f"<td>{row['source_accuracy']:.3f}</td>"
        f"<td>{row['pass_rate']:.3f}</td>"
        f"<td>{row.get('average_prompt_chars', 0):.0f}</td>"
        f"<td>{row.get('average_prompt_est_tokens', 0):.0f}</td>"
        f"<td>{row.get('average_answer_chars', 0):.0f}</td>"
        f"<td>{row['average_latency_seconds']:.2f}s</td>"
        "</tr>"
        for row in summaries
    )
    case_rows = "".join(
        "<tr>"
        f"<td>{result.get('axis_value', result['context_size'])}</td>"
        f"<td>{escape(result['id'])}</td>"
        f"<td>{escape(result['category'])}</td>"
        f"<td>{escape(str(result.get('route') or ''))}</td>"
        f"<td>{escape(str(result.get('long_document_mode') or ''))}</td>"
        f"<td>{result['fact_score']:.3f}</td>"
        f"<td>{result['source_score']:.3f}</td>"
        f"<td>{'PASS' if result['passed'] else 'FAIL'}</td>"
        f"<td>{result.get('prompt_chunk_count', 0)}</td>"
        f"<td>{result.get('prompt_context_chars', 0)}</td>"
        f"<td>{result.get('prompt_context_est_tokens', 0)}</td>"
        f"<td>{result.get('answer_chars', 0)}</td>"
        f"<td>{escape(', '.join(str(item) for item in result['missing_facts']))}</td>"
        f"<td>{result['latency_seconds']:.2f}s</td>"
        "</tr>"
        for result in results
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Context Size Answer Accuracy</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    section {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin: 16px 0; }}
    svg {{ width: 100%; max-width: 900px; height: auto; }}
    rect {{ fill: #4b5563; }}
    circle {{ fill: #374151; }}
    .line {{ stroke: #374151; stroke-width: 3; }}
    .axis {{ stroke: #6b7280; stroke-width: 1; }}
    .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
    .label {{ fill: #111827; font-size: 13px; }}
    .value {{ fill: #4b5563; font-size: 12px; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }}
    .charts section {{ margin: 0; }}
  </style>
</head>
<body>
  <h1>{escape(axis_label)} Answer Accuracy</h1>
  <p>Each QA case was executed through real LLM answer generation, measuring expected fact coverage, source hit rate, and latency.</p>
  <div class="charts">
    {latency_line_chart(summaries, axis_label)}
    {line_chart(summaries, "average_prompt_chars", "Average prompt context chars", axis_label=axis_label, precision=0)}
    {line_chart(summaries, "answer_accuracy", "Answer fact accuracy", axis_label=axis_label, precision=3)}
    {line_chart(summaries, "source_accuracy", "Source accuracy", axis_label=axis_label, precision=3)}
  </div>
  {render_correlation_table(summaries)}
  {bar_chart("Pass Rate", summaries, "pass_rate")}
  <h2>Summary</h2>
  <table><thead><tr><th>{escape(axis_label)}</th><th>Answer accuracy</th><th>Source accuracy</th><th>Pass rate</th><th>Avg prompt chars</th><th>Avg prompt est tokens</th><th>Avg answer chars</th><th>Avg latency</th></tr></thead><tbody>{summary_rows}</tbody></table>
  <h2>Case Results</h2>
  <table><thead><tr><th>{escape(axis_label)}</th><th>Case</th><th>Category</th><th>Route</th><th>Long doc mode</th><th>Fact score</th><th>Source score</th><th>Pass</th><th>Prompt chunks</th><th>Prompt chars</th><th>Prompt est tokens</th><th>Answer chars</th><th>Missing facts</th><th>Latency</th></tr></thead><tbody>{case_rows}</tbody></table>
</body>
</html>
"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


async def async_main(args):
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    load_dotenv(project_root / ".env.local", override=True)
    cases = load_cases(args.cases)
    context_sizes = [int(value.strip()) for value in args.context_sizes.split(",") if value.strip()]
    chunk_counts = [int(value.strip()) for value in args.chunk_counts.split(",") if value.strip()] if args.chunk_counts else None
    results = await evaluate(cases, context_sizes=context_sizes, chunk_counts=chunk_counts)
    summaries = summarize(results)
    render_report(summaries, results, args.output)
    print(json.dumps({"summary": summaries, "output": str(args.output)}, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate answer accuracy across RAG context sizes")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--context-sizes", default="4000,8000,12000,18000,32768")
    parser.add_argument("--chunk-counts", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
