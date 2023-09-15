from typing import Any, Dict, List, Optional
from ray import serve
from neural_search.qdrant import NeuralSearch

@serve.deployment(
    name='NeuralSearch',
    version='1.0',
    init_kwargs=dict(collection_name='faces',
                     url='0.0.0.0',
                     port=6333,
                     grpc_port=6334,
                     prefer_grpc=True,
                     limit=2,
                     score_threshold=0.9,
                     https=False),
    user_config=dict(max_batch_size=4,
                     batch_wait_timeout_s=0.1),
    ray_actor_options={'num_gpus': 0.0,
                       'num_cpus': 0.2,
                       'memory': 600,
                       'runtime_env':{}},
                        health_check_period_s=10,
                        health_check_timeout_s=30,
                        autoscaling_config={'min_replicas': 1,
                                            'max_replicas': 2,
                                            'initial_replicas': 1,},
                        #downscale_delay_s=600,
                        #upscale_delay_s=30
)
class NeuralSearchDeployment(NeuralSearch):

    def __init__(
        self,
        collection_name: str,
        url: str,
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        limit: int = 1,
        score_threshold: float = 0.9,
        https: Optional[bool] = None,
    ):
        super().__init__(url=url,
                         port=port,
                         grpc_port=grpc_port,
                         prefer_grpc=prefer_grpc,
                         https=https)
        self.collection_name = collection_name
        self.limit = limit
        self.score_threshold = score_threshold

    def reconfigure(self, config: Dict) -> None:
        self._handle_batch.set_max_batch_size(
            config.get('max_batch_size', 4))
        self._handle_batch.set_batch_wait_timeout_s(
            config.get('batch_wait_timeout_s', 0.1))
        self.collection_name = config.get('collection_name', 'faces')
        self.limit = config.get('limit', 1)
        self.score_threshold = config.get('score_threshold', 0.9)

    @serve.batch(max_batch_size=4, batch_wait_timeout_s=0.1)
    async def _handle_batch(
        self,
        embeddings: List[List[float]]
    ) -> List[List[Dict[str, Any]]]:

        return await self.search_batch(
            collection_name=self.collection_name,
            embeddings=embeddings,
            limit=self.limit,
            score_threshold=self.score_threshold
        )

    async def search(self, embedding: List[float]) -> List[Dict[str, Any]]:
        return await self._handle_batch(embedding)