import os
from typing import Any, Dict, List
from ray import serve
from neural_search.qdrant import NeuralSearch


@serve.deployment(
    name="NeuralSearch",
    user_config=dict(
        max_batch_size=4, batch_wait_timeout_s=0.1, score_threshold=0.9, top_k=1
    ),
    ray_actor_options={
        "num_gpus": 0.0,
        "num_cpus": 0.2,
        "memory": 600,
        "runtime_env": {
            "image": "vilsonrodrigues/qdrant-ray:2.7.0-py310",
            # "run_options": ["--cap-drop SYS_ADMIN","--log-level=debug"]
        },
    },
    health_check_period_s=10,
    health_check_timeout_s=30,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 2,
        "initial_replicas": 1,
    },
)
class NeuralSearchDeployment(NeuralSearch):
    def __init__(self):
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", default="faces")

        url = os.getenv("QDRANT_URL", default="0.0.0.0")

        port = int(os.getenv("QDRANT_PORT", default="6333"))

        grpc_port = int(os.getenv("QDRANT_GRPC_PORT", default="6334"))

        self.limit = int(os.getenv("QDRANT_TOP_K", default="1"))

        self.score_threshold = float(os.getenv("QDRANT_SCORE_THRESHOLD", default="0.9"))

        prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", default="True")

        https = os.getenv("QDRANT_HTTPS", default="False")

        super().__init__(
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=(prefer_grpc == "True"),
            https=(https == "True"),
        )

    def reconfigure(self, config: Dict) -> None:
        self._handle_batch.set_max_batch_size(config.get("max_batch_size", 4))

        self._handle_batch.set_batch_wait_timeout_s(
            config.get("batch_wait_timeout_s", 0.1)
        )

        self.score_threshold = config.get("score_threshold", 0.9)

        self.limit = config.get("top_k", 1)

    @serve.batch(max_batch_size=4, batch_wait_timeout_s=0.1)
    async def _handle_batch(
        self, embeddings: List[List[float]]
    ) -> List[List[Dict[str, Any]]]:
        return await self.search_batch(
            collection_name=self.collection_name,
            embeddings=embeddings,
            limit=self.limit,
            score_threshold=self.score_threshold,
        )

    async def search(self, embedding: List[float]) -> List[Dict[str, Any]]:
        return await self._handle_batch(embedding)
