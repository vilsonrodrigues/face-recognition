from ray.serve.drivers import DAGDriver
from ray.serve.deployment_graph import InputNode

from handlers.retrieval_logic import PipelineRetrievalLogic
from http_adapters.uploadfile_ndarray import UploadFileToNdarray
from predictors.online.mobile_facenet_ort import MobileFaceNetDeploymentORT
from predictors.online.ultra_light_ort import UltraLightDeploymentORT
from processing.online.face_postp_deployment import FacePostProcessingDeployment
from neural_search.online.ns_deployment import NeuralSearchDeployment

uploadfile = UploadFileToNdarray.bind()

face_det = UltraLightDeploymentORT.bind()

face_emb = MobileFaceNetDeploymentORT.bind()

face_postp = FacePostProcessingDeployment.bind()

ns = NeuralSearchDeployment.bind()

retrieval = PipelineRetrievalLogic.bind(
    face_emb,
    face_postp,
    ns,
)

with InputNode() as image_bytes:

    image_np = uploadfile.base64_to_ndarray.bind(image_bytes)

    relative_boxes = face_det.predict.bind(image_np)

    output = retrieval.route.bind(relative_boxes, image_np)

graph = (DAGDriver
         .options(name='driver',
                  route_prefix='/',
                  ray_actor_options={
                    'num_gpus': 0.0,
                    'num_cpus': 0.5,
                    'memory': 600,
                    'runtime_env':
                        {'image': 'rayproject/ray:2.7.0-py310'}
                  }
                   health_check_period_s=10,
                   health_check_timeout_s=30,
                   autoscaling_config={
                       'min_replicas': 1,
                       'max_replicas': 2,
                       'initial_replicas': 1,
                   },
         )
         .bind(output))