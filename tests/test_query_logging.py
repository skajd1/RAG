import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import _paths  # noqa: F401
from query_logging import QueryLogWriter


class QueryLogWriterTests(unittest.TestCase):
    def test_writes_raw_question_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = QueryLogWriter(enabled=True, directory=tmp, include_raw_text=True, retention_days=30)

            writer.write({"question": "사내 전결 규정 알려줘", "route": "exact_lookup"})

            files = list(Path(tmp).glob("*.jsonl"))
            self.assertEqual(len(files), 1)
            record = json.loads(files[0].read_text(encoding="utf-8").strip())
            self.assertEqual(record["question"], "사내 전결 규정 알려줘")
            self.assertEqual(record["route"], "exact_lookup")

    def test_hashes_question_when_raw_text_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = QueryLogWriter(enabled=True, directory=tmp, include_raw_text=False, retention_days=30)

            writer.write({"question": "secret question", "route": "fallback"})

            record = json.loads(next(Path(tmp).glob("*.jsonl")).read_text(encoding="utf-8").strip())
            self.assertNotIn("question", record)
            self.assertIn("question_hash", record)

    def test_disabled_writer_does_not_create_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "query"
            writer = QueryLogWriter(enabled=False, directory=target, include_raw_text=True, retention_days=30)

            writer.write({"question": "ignored"})

            self.assertFalse(target.exists())

    def test_retention_removes_old_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "2026-01-01.jsonl"
            old.write_text("{}\n", encoding="utf-8")
            very_old = datetime.now(timezone.utc) - timedelta(days=40)
            os.utime(old, (very_old.timestamp(), very_old.timestamp()))
            writer = QueryLogWriter(enabled=True, directory=tmp, include_raw_text=True, retention_days=30)

            writer.write({"question": "new"})

            self.assertFalse(old.exists())
