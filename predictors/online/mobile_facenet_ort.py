import os
from typing import Dict, List

from PIL import Image
import numpy as np
from ray import serve

from face_embedding.mobile_facenet_ort import MobileFaceNetORT

@serve.deployment(
    name='MobileFaceNetORT',
    user_config=dict(max_batch_size=4,
                     batch_wait_timeout_s=0.1),
    ray_actor_options={'num_gpus': 0.0,
                       'num_cpus': 1.0,
                       'memory': 2048,
                       'runtime_env':
                            {'image': 'vilsonrodrigues/onnxruntime-openvino-ray:2.7.0-py310',
                            #"run_options": ["--cap-drop SYS_ADMIN","--log-level=debug"]
                             }
                       },
    health_check_period_s=10,
    health_check_timeout_s=30,
    autoscaling_config={'min_replicas': 1,
                        'max_replicas': 4,
                        'initial_replicas': 1,}
)
class MobileFaceNetORTDeployment(MobileFaceNetORT):

    def __init__(self):

        # getenvs

        backend = os.getenv('BACKEND_MOB_FACENET', default='openvino')

        model_path = os.getenv('MODEL_PATH_MOB_FACENET',
                               default='models/mobilefacenet_prep.onnx')

        warmup_rounds = int(os.getenv('WARMUP_ROUNDS', default='3'))

        # format

        self.backend = None if backend == 'None' else backend

        super().__init__(model_path, self.backend)

        self._warmup(warmup_rounds)

    def _warmup(self, warmup_rounds: int) -> None:

        for _ in range(warmup_rounds):

            self._predict(np.random.rand(253, 413, 3).astype(np.uint32))

    def reconfigure(self, config: Dict) -> None:

        self._handle_batch.set_max_batch_size(
            config.get('max_batch_size', 4))

        self._handle_batch.set_batch_wait_timeout_s(
            config.get('batch_wait_timeout_s', 0.1))

    @serve.batch(max_batch_size=4, batch_wait_timeout_s=0.1)
    async def _handle_batch(
        self,
        input_batch: List[np.ndarray]
    ) -> List[List[float]]:

        batch = [np.expand_dims(np.array(Image.fromarray(image).resize((112, 112))), axis=0) for image in input_batch]

        concatenated_batch = np.concatenate(batch, axis=0)

        embeddings = self._predict(concatenated_batch)[0]

        embeddings = embeddings.tolist()

        return embeddings

    async def predict(self, input_data: np.ndarray) -> List[float]:
        return await self._handle_batch(input_data)