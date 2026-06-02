import argparse
import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path


def load_records(input_dir: Path):
    records = []
    for path in sorted(input_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def average(values):
    values = list(values)
    return sum(values) / len(values) if values else 0


def render(records):
    route_counts = Counter(record.get("route", "unknown") for record in records)
    question_counts = Counter(
        record.get("question") or record.get("question_hash") or "(hidden)"
        for record in records
    )
    finish_counts = Counter(record.get("finish_reason", "unknown") for record in records)
    latency_by_route = defaultdict(list)
    chunks_by_route = defaultdict(list)
    for record in records:
        route = record.get("route", "unknown")
        latency_by_route[route].append(float(record.get("total_seconds", 0) or 0))
        chunks_by_route[route].append(int(record.get("prompt_chunks", 0) or 0))

    def count_rows(counter):
        return "".join(
            f"<tr><td>{escape(str(key))}</td><td>{value}</td></tr>"
            for key, value in counter.most_common(20)
        )

    route_rows = "".join(
        "<tr>"
        f"<td>{escape(route)}</td>"
        f"<td>{route_counts[route]}</td>"
        f"<td>{average(latency_by_route[route]):.2f}s</td>"
        f"<td>{average(chunks_by_route[route]):.1f}</td>"
        "</tr>"
        for route in sorted(route_counts)
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Query Log Analysis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 32px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>Query Log Analysis</h1>
  <p>Total records: {len(records)}</p>
  <h2>Routes</h2>
  <table><thead><tr><th>Route</th><th>Count</th><th>Avg Latency</th><th>Avg Prompt Chunks</th></tr></thead><tbody>{route_rows}</tbody></table>
  <h2>Top Questions</h2>
  <table><thead><tr><th>Question or Hash</th><th>Count</th></tr></thead><tbody>{count_rows(question_counts)}</tbody></table>
  <h2>Finish Reasons</h2>
  <table><thead><tr><th>Reason</th><th>Count</th></tr></thead><tbody>{count_rows(finish_counts)}</tbody></table>
</body>
</html>"""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze MetsaBrain query logs")
    parser.add_argument("--input", type=Path, default=Path("runtime-logs/query"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    records = load_records(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(records), encoding="utf-8")
    print(json.dumps({"records": len(records), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
