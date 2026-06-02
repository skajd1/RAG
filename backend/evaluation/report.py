import json
from html import escape
from pathlib import Path


def load_golden_cases(path):
    cases = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            if "relevant_pages" not in case and "relevance_grades" not in case:
                raise ValueError(
                    f"golden case line {line_number} requires relevant_pages or relevance_grades"
                )
            case.setdefault("history", [])
            case.setdefault("forbidden_pages", [])
            case.setdefault("expected_facts", [])
            cases.append(case)
    return cases


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple)):
        return "<br>".join(escape(str(item)) for item in value) or "-"
    return escape(str(value))


def _table(headers, rows):
    heading = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_format_value(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table>"


def _bar_chart(title, rows, chart_id, width=720, row_height=34):
    if not rows:
        return ""
    max_value = max([float(row[1] or 0) for row in rows] + [1.0])
    height = 42 + row_height * len(rows)
    bars = []
    for index, (label, value, note) in enumerate(rows):
        value = float(value or 0)
        y = 32 + index * row_height
        bar_width = int((width - 220) * min(value / max_value, 1.0))
        bars.append(
            f'<text x="0" y="{y + 16}" class="chart-label">{escape(str(label))}</text>'
            f'<rect x="180" y="{y}" width="{bar_width}" height="20" rx="4"></rect>'
            f'<text x="{190 + bar_width}" y="{y + 15}" class="chart-value">{value:.3f} {escape(str(note or ""))}</text>'
        )
    return (
        f'<section class="chart" data-chart="{escape(chart_id)}">'
        f"<h3>{escape(title)}</h3>"
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        + "".join(bars)
        + "</svg></section>"
    )


def _summary_target_rows(summary, targets):
    metric_labels = {
        "hit_at_k": "Hit@K",
        "recall_at_k": "Recall@K",
        "precision_at_k": "Precision@K",
        "mrr": "MRR",
        "ndcg_at_k": "nDCG@K",
        "forbidden_scope_rate": "Forbidden scope",
    }
    rows = []
    for metric, label in metric_labels.items():
        if metric in summary:
            target = targets.get(metric)
            note = f"/ target {target:.2f}" if isinstance(target, (int, float)) else ""
            rows.append((label, summary[metric], note))
    return rows


def _category_rows(category_summaries, metric="recall_at_k"):
    return [
        (category, values.get(metric, 0.0), metric)
        for category, values in sorted(category_summaries.items())
        if metric in values
    ]


def _baseline_rows(summary, baseline_summary):
    rows = []
    for metric in ("hit_at_k", "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k"):
        if metric in summary and metric in baseline_summary:
            delta = float(summary[metric] or 0) - float(baseline_summary[metric] or 0)
            rows.append((metric, delta, f"current {summary[metric]:.3f} / baseline {baseline_summary[metric]:.3f}"))
    return rows


def render_html_report(
    summary,
    category_summaries,
    case_results,
    title="RAG Retrieval Quality",
    targets=None,
    baseline_summary=None,
    baseline_case_results=None,
):
    targets = targets or {}
    baseline_summary = baseline_summary or {}
    baseline_case_results = baseline_case_results or []
    summary_table = _table(["Metric", "Value"], summary.items())

    category_metrics = sorted(
        {metric for values in category_summaries.values() for metric in values}
    )
    category_rows = [
        [category, *(values.get(metric, "") for metric in category_metrics)]
        for category, values in sorted(category_summaries.items())
    ]
    category_table = _table(["Category", *category_metrics], category_rows)

    case_rows = [
        [
            case.get("id", ""),
            case.get("category", ""),
            case.get("query", ""),
            case.get("resolved_query", ""),
            case.get("scope_changed", ""),
            case.get("expected_query_match", ""),
            case.get("route", ""),
            case.get("route_reason", ""),
            (case.get("long_document_stats") or {}).get("long_document_mode", ""),
            case.get("latency_ms", ""),
            case.get("retrieved_pages", []),
            case.get("hit_at_k", ""),
            case.get("recall_at_k", ""),
            case.get("precision_at_k", ""),
            case.get("reciprocal_rank", ""),
            case.get("ndcg_at_k", ""),
            case.get("forbidden_pages", []),
        ]
        for case in case_results
    ]
    case_table = _table(
        [
            "ID",
            "Category",
            "Query",
            "Resolved query",
            "Scope changed",
            "Expected query match",
            "Route",
            "Route reason",
            "Long doc mode",
            "Latency (ms)",
            "Retrieved pages",
            "Hit@k",
            "Recall@k",
            "Precision@k",
            "Reciprocal rank",
            "nDCG@k",
            "Forbidden pages",
        ],
        case_rows,
    )
    charts = "\n".join(
        chart
        for chart in [
            _bar_chart("Summary Metric Targets", _summary_target_rows(summary, targets), "summary-targets"),
            _bar_chart("Category Metrics", _category_rows(category_summaries), "category-metrics"),
            _bar_chart("Baseline vs Current", _baseline_rows(summary, baseline_summary), "baseline-current"),
        ]
        if chart
    )

    baseline_table = ""
    if baseline_case_results:
        baseline_table = _table(
            ["ID", "Hit@k", "Recall@k", "Precision@k", "nDCG@k"],
            [
                [
                    case.get("id", ""),
                    case.get("hit_at_k", ""),
                    case.get("recall_at_k", ""),
                    case.get("precision_at_k", ""),
                    case.get("ndcg_at_k", ""),
                ]
                for case in baseline_case_results
            ],
        )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f3f3; }}
    .chart {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 18px 0; }}
    .chart svg {{ width: 100%; max-width: 920px; height: auto; }}
    .chart rect {{ fill: #4b5563; }}
    .chart-label {{ fill: #222; font-size: 13px; }}
    .chart-value {{ fill: #555; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  {charts}
  <h2>Summary</h2>
  {summary_table}
  {"<h2>Baseline Cases</h2>" + baseline_table if baseline_table else ""}
  <h2>Categories</h2>
  {category_table}
  <h2>Cases</h2>
  {case_table}
</body>
</html>
"""
