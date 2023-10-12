from typing import Any, Dict, List, Optional, Union
from qdrant_client import models, QdrantClient

class NeuralSearch:

    def __init__(
        self,
        location: Union[str, None] = None, #":memory:"
        path: Optional[Union[str, None]] = None, #"path/to/db",
        port: Optional[int] = 6333,
        grpc_port: Optional[int] = 6334,
        prefer_grpc: Optional[Union[bool, None]] = False,
        url: Optional[Union[str, None]] = None,
        https: Optional[bool] = None,
    ):
        self.client = QdrantClient(location=location,
                                   url=url,
                                   path=path,
                                   port=port,
                                   grpc_port=grpc_port,
                                   https=https,
                                   prefer_grpc=prefer_grpc)

    def get_collections(self):
        return self.client.get_collections()

    def count(self, collection_name: str) -> int:
        return self.client.count(collection_name).count

    def scroll(
        self,
        collection_name: str,
        limit: int,
        with_vectors: Optional[bool] = False,
        with_payloads: Optional[bool] = True
    ) -> List:

        return self.client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_vectors=with_vectors,
            with_payloads=with_payloads
        )

    def create_collection(
        self,
        collection_name: str,
        embedding_dim: int,
        distance: Optional[models.Distance] = models.Distance.COSINE
    ) -> bool:
        """
        Create a collection in Qdrant
        Args:
            collection_name
            embedding_dim: embedding dimension
            distance: metric to calculate distance
        """
        return self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_dim,
                    distance=distance
                )
        )

    def insert(
        self,
        collection_name: str,
        embeddings: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[int]] = None,
        batch_size: Optional[int] = 256
    ) -> bool:

        return self.client.upload_collection(
            collection_name=collection_name,
            vectors=embeddings,
            payload=payloads,
            ids=ids,
            batch_size=batch_size
        )

    def search(
        self,
        collection_name: str,
        embedding: List[float],
        limit: Optional[int] = 3, # top k
        score_threshold: Optional[float] = 0.3,
    ) -> List[Dict[str, Any]]:

        search_result = self.client.search(
            collection_name=collection_name,
            limit=limit,
            query_vector=embedding,
            score_threshold=score_threshold,
            with_payload=True
        )

        payloads = [hit.payload for hit in search_result]

        return payloads

    def search_batch(
        self,
        collection_name: str,
        embeddings: List[List[float]],
        limit: Optional[int] = 3, # top k
        score_threshold: Optional[float] = 0.3,
    ) -> List[List[Dict[str, Any]]]:

        search_queries = [
            models.SearchRequest(
                vector=embedding,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )
            for embedding in embeddings
        ]

        search_results = self.client.search_batch(
                            collection_name=collection_name,
                            requests=search_queries,
                        )

        payloads = [[hit.payload for hit in sublist] for sublist in search_results]

        return payloads

    def retrieve(
        self,
        collection_name: str,
        ids: List[int],
        with_vectors: Optional[bool] = False,
    ) -> List:
        """ Retriver based-on ids """
        return self.client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_vectors=with_vectors
        )