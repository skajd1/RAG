import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


class QueryLogWriter:
    def __init__(
        self,
        enabled: bool | None = None,
        directory: str | Path | None = None,
        include_raw_text: bool | None = None,
        retention_days: int | None = None,
    ):
        self.enabled = self.env_bool("QUERY_LOG_ENABLED", True) if enabled is None else enabled
        self.directory = Path(directory or os.getenv("QUERY_LOG_DIR", "runtime-logs/query"))
        self.include_raw_text = (
            self.env_bool("QUERY_LOG_INCLUDE_RAW_TEXT", True)
            if include_raw_text is None
            else include_raw_text
        )
        self.retention_days = (
            int(os.getenv("QUERY_LOG_RETENTION_DAYS", "30"))
            if retention_days is None
            else retention_days
        )

    def env_bool(self, key: str, default: bool) -> bool:
        raw = os.getenv(key)
        if raw is None:
            return default
        return raw.lower() in {"1", "true", "yes", "on"}

    def write(self, record: dict):
        if not self.enabled:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.cleanup_old_files()
            path = self.directory / f"{datetime.now().date().isoformat()}.jsonl"
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(self.normalize_record(record), ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[QUERY_LOG] write failed: {exc}")

    def normalize_record(self, record: dict) -> dict:
        normalized = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            **record,
        }
        question = normalized.get("question")
        if not self.include_raw_text and question is not None:
            normalized["question_hash"] = hashlib.sha256(str(question).encode("utf-8")).hexdigest()
            normalized.pop("question", None)
        return normalized

    def cleanup_old_files(self):
        if self.retention_days <= 0:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        for path in self.directory.glob("*.jsonl"):
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if modified < cutoff:
                path.unlink(missing_ok=True)
