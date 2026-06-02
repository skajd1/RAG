import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import _paths  # noqa: F401
from backend.database import VectorDB


class VectorDBInitializationTests(unittest.TestCase):
    def test_dimension_mismatch_does_not_delete_existing_collection(self):
        db = VectorDB.__new__(VectorDB)
        db.collection_name = "confluence_docs"
        db.client = Mock()
        db.client.get_collections.return_value = SimpleNamespace(
            collections=[SimpleNamespace(name="confluence_docs")]
        )
        db.client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=1536)))
        )

        with patch.object(db, "_create_new_collection") as create_collection:
            with self.assertRaisesRegex(RuntimeError, "Vector dimension mismatch"):
                db.init_collection(vector_size=4096)

        db.client.delete_collection.assert_not_called()
        create_collection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
