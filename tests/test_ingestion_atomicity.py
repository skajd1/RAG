import unittest
from unittest.mock import patch

from langchain_core.documents import Document

import _paths  # noqa: F401
from backend.ingestion import save_source_documents


class SaveSourceDocumentsTests(unittest.TestCase):
    def test_replacement_deletes_stale_documents_after_new_documents_are_saved(self):
        events = []
        documents = [Document(page_content="new content", metadata={"space": "TEAM"})]

        with (
            patch("backend.ingestion.QdrantVectorStore.from_documents", side_effect=lambda *args, **kwargs: events.append("save")),
            patch("backend.ingestion.delete_existing_source_space", side_effect=lambda *args, **kwargs: events.append("delete")) as delete,
        ):
            run_id = save_source_documents("http://qdrant", "confluence", "TEAM", documents, object(), True)

        self.assertEqual(events, ["save", "delete"])
        self.assertEqual(documents[0].metadata["ingest_run_id"], run_id)
        delete.assert_called_once_with(
            "http://qdrant",
            "confluence",
            "TEAM",
            keep_ingest_run_id=run_id,
        )

    def test_failed_save_cleans_only_documents_from_failed_run(self):
        documents = [Document(page_content="new content", metadata={"space": "TEAM"})]

        with (
            patch("backend.ingestion.QdrantVectorStore.from_documents", side_effect=RuntimeError("embedding failed")),
            patch("backend.ingestion.delete_existing_source_space") as delete,
        ):
            with self.assertRaisesRegex(RuntimeError, "embedding failed"):
                save_source_documents("http://qdrant", "confluence", "TEAM", documents, object(), True)

        run_id = documents[0].metadata["ingest_run_id"]
        delete.assert_called_once_with(
            "http://qdrant",
            "confluence",
            "TEAM",
            only_ingest_run_id=run_id,
        )


if __name__ == "__main__":
    unittest.main()
