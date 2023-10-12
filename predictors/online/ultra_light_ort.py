import os
from typing import Dict, List

import numpy as np
from PIL import Image
from ray import serve

from face_detection.ultra_light_ort import UltraLightORT

@serve.deployment(
    name='UltraLightORT',
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
                        'initial_replicas': 1,},
)
class UltraLightORTDeployment(UltraLightORT):

    def __init__(self):

        # getenvs

        apply_resize = os.getenv('APPLY_RESIZE', default='True')

        backend = os.getenv('BACKEND_ULTRA_LIGHT', default='openvino')

        model_path = os.getenv('MODEL_PATH_ULTRA_LIGHT',
                      default='models/ultralight_RBF_320_prep_nms.onnx')

        warmup_rounds = int(os.getenv('WARMUP_ROUNDS', default='3'))

        # format

        self.apply_resize = False if apply_resize == 'False' else True

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
    ) -> List[np.ndarray]:

        # if image input not have a same dimensions
        if self.apply_resize:

            batch = [np.expand_dims(np.array(Image.fromarray(image).resize((320, 240))), axis=0) for image in input_batch]

        else:

            batch = [np.expand_dims(image, axis=0) for image in input_batch]

        concatenated_batch = np.concatenate(batch, axis=0)

        boxes, batch_indices = self._predict(concatenated_batch)

        # if no face is detected in any batch
        if len(batch_indices) == 0:

            # add empty arrays for each batch
            boxes_by_batch = [np.array([]) for _ in range(len(input_batch))]

        else:

            boxes_by_batch = self.split_boxes_by_batch(boxes, batch_indices)

        return boxes_by_batch

    async def predict(
        self,
        input_data: np.ndarray
    ) -> List[np.ndarray]:
        return await self._handle_batch(input_data)