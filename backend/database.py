import os
import time

from qdrant_client import QdrantClient
from qdrant_client.http import models


class VectorDB:
    def __init__(self):
        self.host = os.getenv("QDRANT_HOST", "http://localhost:6333")
        print(f"Connecting to Qdrant at: {self.host}")

        for i in range(5):
            try:
                self.client = QdrantClient(url=self.host)
                self.client.get_collections()
                print("Successfully connected to Qdrant.")
                break
            except Exception as e:
                print(f"Connection attempt {i + 1} failed. Retrying in 2s...")
                if i == 4:
                    raise e
                time.sleep(2)

        self.collection_name = "confluence_docs"

    def init_collection(self, vector_size: int = 1536):
        try:
            collections = self.client.get_collections().collections
            collection_info = next((c for c in collections if c.name == self.collection_name), None)

            if collection_info:
                current_info = self.client.get_collection(self.collection_name)
                current_size = current_info.config.params.vectors.size

                if current_size != vector_size:
                    raise RuntimeError(
                        f"Vector dimension mismatch for collection '{self.collection_name}': "
                        f"{current_size} != {vector_size}. Refusing to delete existing data automatically."
                    )
                else:
                    print(f"Collection '{self.collection_name}' already exists with correct size.")
            else:
                self._create_new_collection(vector_size)

        except Exception as e:
            print(f"Error initializing collection: {e}")
            raise e

    def _create_new_collection(self, vector_size):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Collection '{self.collection_name}' created with size {vector_size}.")

    def get_client(self):
        return self.client
