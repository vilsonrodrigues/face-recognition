import os
from typing import List
import numpy as np
import onnxruntime as ort
from base.base_models import ModelBaseClass

class ONNXRuntimeModel(ModelBaseClass):

    """
    Base class for inference in cpu or device

    methods:
        __init__(self, MODEL_PATH: str) -> None

        cuda_is_avaliable() -> bool

        _load_model(self) -> None

        _predict(self, input_data: np.ndarray) -> List[ort.OrtValue]

        __call__(self, input_data: np.ndarray) -> List[ort.OrtValue]
    """

    def __init__(self, MODEL_PATH: str) -> None:
        if os.path.exists(MODEL_PATH):
            self.MODEL_PATH = MODEL_PATH
            self._load_model()
        else:
            raise ValueError(f'Model not exists in {MODEL_PATH}')

    @staticmethod
    def cuda_is_avaliable() -> bool:
        if ort.get_device() == 'GPU':
            return True
        else:
            return False

    def _load_model(self) -> None:
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 4

        self.EP = ['CPUExecutionProvider']

        if ONNXRuntimeModel.cuda_is_avaliable():
            self.EP.insert(0, 'CUDAExecutionProvider')
            # model optimization
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.execution_order = ort.ExecutionOrder.PRIORITY_BASED
            sess_options.optimized_model_filepath = self.MODEL_PATH.replace('.','_opt.')

        self._model = ort.InferenceSession(self.MODEL_PATH,
                                           providers=self.EP,
                                           sess_options=sess_options,
                                           )
        self._input_name = self._model.get_inputs()[0].name

    def _predict(self, input_data: np.ndarray) -> List[ort.OrtValue]:

        if len(input_data.shape) == 3:
            # add batch dim
            input_data = np.expand_dims(input_data, axis=0)

        ortvalue = ort.OrtValue.ortvalue_from_numpy(input_data.astype(np.float32))

        # inference on gpu
        if ONNXRuntimeModel.cuda_is_avaliable():

            io_binding = self._model.io_binding()

            io_binding.bind_input(name=self._input_name,
                                  device_type=ortvalue.device_name(),
                                  device_id=0,
                                  element_type=np.float32,
                                  shape=ortvalue.shape(),
                                  buffer_ptr=ortvalue.data_ptr())

            io_binding.bind_output(None)

            self._model.run_with_iobinding(io_binding)

            return io_binding.copy_outputs_to_cpu()

        # inference on cpu
        else:
            return self._model.run_with_ort_values(None, {self._input_name: ortvalue})

    def __call__(self, input_data: np.ndarray) -> List[ort.OrtValue]:
        return self._predict(input_data)