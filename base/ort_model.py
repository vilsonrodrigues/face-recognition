import os
from typing import List, Optional

import numpy as np
import onnxruntime as ort

from base.base_models import ModelBaseClass


class ONNXRuntimeModel(ModelBaseClass):

    """
    Base class for inference in cpu or backend
    """

    def __init__(self, model_path: str, backend: Optional[str] = None):
        """
        Args:
            model_path
            backend: backend supported are `cuda` or `openvino`.
                     For OpenVINO install `onnxruntime-openvino` package.
                     For CUDA use `onnxruntime-gpu`.
        """
        if os.path.exists(model_path):
            self.model_path = model_path
            self.backend = backend
            self._load_model()
        else:
            raise ValueError(f"Model not exists in {model_path}")

    @staticmethod
    def cuda_is_avaliable() -> bool:
        if ort.get_device() == "GPU":
            return True
        else:
            return False

    def _load_model(self) -> None:
        sess_options = ort.SessionOptions()

        sess_options.log_severity_level = 4

        self.EP = ["CPUExecutionProvider"]

        provider_options = []

        if self.backend == "openvino":
            self.EP = ["OpenVINOExecutionProvider"]

            # get folder from original model path
            path_to_find = self.model_path.split("/")[-2]
            end_index = self.model_path.find(path_to_find) + len(path_to_find)
            cache_dir = self.model_path[:end_index]

            provider_options.append(
                {
                    "device_type": "CPU_FP16",
                    "enable_dynamic_shapes": True,
                    "cache_dir": cache_dir,
                }
            )

            # prefer OpenVINO optimizations
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )

        elif self.backend == "cuda":
            if ONNXRuntimeModel.cuda_is_avaliable():
                self.EP.insert(0, "CUDAExecutionProvider")
                # model optimization
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                sess_options.execution_order = ort.ExecutionOrder.PRIORITY_BASED
                sess_options.optimized_model_filepath = self.model_path.replace(
                    ".", "_opt."
                )
            else:
                raise Exception("No CUDA device detected")

        self._model = ort.InferenceSession(
            self.model_path,
            providers=self.EP,
            sess_options=sess_options,
            provider_options=(provider_options if provider_options else None),
        )

        # this code is adapted to work with 1 input
        self._input_name = self._model.get_inputs()[0].name
        # but with multi output
        output_name = []
        for i in range(len(self._model.get_outputs())):
            output_name.append(self._model.get_outputs()[i].name)
        self._output_name = output_name

    def _predict(self, input_data: np.ndarray) -> List[np.ndarray]:
        if len(input_data.shape) == 3:
            # add batch dim
            input_data = np.expand_dims(input_data, axis=0)

        ortvalue = ort.OrtValue.ortvalue_from_numpy(input_data.astype(np.float32))

        # inference on cuda
        if self.backend == "cuda":
            io_binding = self._model.io_binding()

            io_binding.bind_input(
                name=self._input_name,
                device_type=ortvalue.device_name(),
                device_id=0,
                element_type=np.float32,
                shape=ortvalue.shape(),
                buffer_ptr=ortvalue.data_ptr(),
            )

            for output_name in self._output_name:
                io_binding.bind_output(output_name)

            self._model.run_with_iobinding(io_binding)

            outputs = [
                io_binding.copy_output_to_cpu(output_name)
                for output_name in self._output_name
            ]

            return outputs

        # inference on cpu
        else:
            outputs = self._model.run_with_ort_values(
                None, {self._input_name: ortvalue}
            )
            outputs_np = [output.numpy() for output in outputs]
            return outputs_np

    def __call__(self, input_data: np.ndarray) -> List[np.ndarray]:
        return self._predict(input_data)
