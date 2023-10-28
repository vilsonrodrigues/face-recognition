from handlers.retrieval_pipeline import PipelineRetrieval
from predictors.online.mobile_facenet_ort import MobileFaceNetORTDeployment
from predictors.online.ultra_light_ort import UltraLightORTDeployment
from processing.online.face_postp_deployment import FacePostProcessingDeployment
from neural_search.online.ns_deployment import NeuralSearchDeployment

face_det = UltraLightORTDeployment.bind()

face_emb = MobileFaceNetORTDeployment.bind()

face_postp = FacePostProcessingDeployment.bind()

ns = NeuralSearchDeployment.bind()

pipeline = PipelineRetrieval.bind(    
    face_det,
    face_emb,
    face_postp,
    ns,
)
