# A Scalable Face Recognition System

<p align="center">
    <img alt="Static Badge" src="https://img.shields.io/badge/Project_Status-Ready%20For%20Production-29cd37">
    <img alt="GitHub" src="https://img.shields.io/github/license/vilsonrodrigues/face-recognition.svg?color=yellow">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/vilsonrodrigues/face-recognition.svg?color=green">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/vilsonrodrigues/face-recognition.svg?color=rose">
</p>

<h2> Overview </h2> 

The goal this project was to build a lightweight and scalable face recognition system. To solve bottlenecks as serial pre and post processing and slow face matching some strategies were adopted.

Efficient and lightweight models were selected. [Ultra LightWeight](https://github.com//) face detector and [Mobile FaceNet](https://github.com/Linzaerfoamliu/MobileFaceNet) to feature extraction.

Preprocessing steps were fusioned on both models. NMS layers were add to face detector using ONNX. IoU thresh was 0.5 and conf thresh were 0.95. The models graphs are simplified using [ONNX-sim](https://github.com/daquexian/onnx-simplifier).

[ONNX Runtime](https://onnxruntime.ai/) with [OpenVINO](https://docs.openvino.ai/2023.2/home.html) are the inference engine for model execution.

[Qdrant](https://qdrant.tech/) is used as a Vector Search Engine (Vector Database) to efficient face matching and data retrieval.

[Ray](https://ray.io) is a python lib to distributed execution in large-scale. Ray runs on laptop, clouds, Kubernetes or on-promise. In this project, Ray Data ingests and processes images. Ray Serve model composition API join five services to face retrieval. [FastAPI](https://fastapi.tiangolo.com/) is integrate to recive HTTP requests.

<h2> Getting Started </h2> 

<h3> Configure Deps </h3> 

<h4> Kubernetes </h4> 

The recommendation for deploying this service is in Kubernetes via KubeRay. You need Kubernetes and Helm to follow these steps

<h5> Install KubeRay </h5>

KubeRay is way to deploy Ray's applications in Kubernetes clusters

``` shell
chmod +x install_kuberay_deps.sh
```

``` shell
./install_kuberay_deps.sh
```

<h5> Install Prometheus and Grafana (Optional) </h5>

KubeRay project offers support to integrate Prometheus and Grafana. To use, install

``` shell
chmod +x install_prometheus_grafana.sh
```

``` shell
./install_prometheus_grafana.sh
```

Check the installation

``` shell
kubectl get all -n prometheus-system
```

KubeRay exposes a Prometheus metrics endpoint in port 8080. Please check [Ray documentation](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/prometheus-grafana.html) to configure ports to Prometheus and Grafana, and integrate to Ray Dashboard. 

See [how to configure](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/ingress.html) an Ingress for Ray Dashboard.

<h5> Install Qdrant (Optional) </h5>

You can install Qdrant locally via Helm or use Qdrand Cloud

``` shell
chmod +x install_qdrant_helm.sh
```

``` shell
./install_qdrant_helm.sh
```

Qdrant default ports are 6333 to HTTP and 6334 to gRPC

To forward Qdrant's ports execute one of the following commands:

``` shell
export POD_NAME=$(kubectl get pods --namespace default -l "app.kubernetes.io/name=qdrant,app.kubernetes.io/instance=qdrant" -o jsonpath="{.items[0].metadata.name}")
```

If you want to use Qdrant via HTTP execute the following commands in other terminal tab:

``` shell
kubectl --namespace default port-forward $POD_NAME 6333:6333
```

If you want to use Qdrant via gRPC execute the following commands in other terminal tab:

``` shell
kubectl --namespace default port-forward $POD_NAME 6334:6334
```

Via HTTP, you can view the Qdrand dashboard `<your-ip>:6333/dashboard`

<h3> Data Ingestion with Ray Data</h3>

Ray Data is use to datasource consuming and to efficient batch processing from images with `map_batches` API. 

<img src="assets\face-ingestion.png">

Many DB connectors are implemented: MySQL, MSSQL, Postegres, Databricks and Snowflake. In `jobs` directory, some examples were demonstrated of how to develop a Job. In `kubernetes` directory some RayJobs are described.

The embeddings generated are stored in vector search engine (Qdrant).

Attention, RayJob can be not initialized if limits no are available in Kubernetes cluster. Check and adapt for your cluster

``` shell
kubectl apply -f kubernetes/job_lfw.yaml
```

List all RayJob custom resources in the `default` namespace

``` shell
kubectl get rayjob
```

List all RayCluster custom resources in the `default` namespace

``` shell
kubectl get raycluster
```

List all Pods in the `default` namespace. The Pod created by the Kubernetes Job will be terminated after the Kubernetes Job finishes

``` shell
kubectl get pods
```

Check the status of the RayJob. The field `jobStatus` in the RayJob custom resource will be updated to `SUCCEEDED` once the job finishes

``` shell
kubectl get rayjobs.ray.io rayjob-lfw -o json | jq '.status.jobStatus'
```

Check the RayJob logs

``` shell
kubectl logs -l=job-name=rayjob-lfw
```

Delete RayJob

``` shell
kubectl delete -f job_lfw.yaml 
```

<h3> Deploy Online API with RayService </h3>

<img src="assets\face-retrieval.png">

The online service will information retrieval based-on an input image. Check API routers:

| Method | Router          | Data Type  |
|--------|----------------|---------------|
| POST   | /base64        | String        |
| POST   | /uploadfile    | Bytes    |
| GET    | /              | -             |

The API output is a JSON with two fields: `payloads` and `boxes`.

CORS are configured by default. JWT tokens are supporteds. You need configure two envs: `AUTH_ALGORITHM` and `AUTH_SECRET_KEY`. Check RayService manifest to configure Qdrant params and more.

``` shell
kubectl apply -f kubernetes/face-recog-svc.yaml
```

List all RayService custom resources in the `default` namespace

``` shell
kubectl get rayservice
```

List all RayCluster custom resources in the `default` namespace

``` shell
kubectl get raycluster
```

List all Ray Pods in the `default` namespace

``` shell
kubectl get pods -l=ray.io/is-ray-node=yes
```

List services in the `default` namespace

``` shell
kubectl get services
```
Check the status of the RayService

``` shell
kubectl describe rayservices rayservice-face-recog
```

Expose Ray Serve port

``` shell
kubectl port-forward svc/rayservice-face-recog-serve-svc --address 0.0.0.0 8000:8000
```           

Expose Ray Dashboard port

``` shell
kubectl port-forward svc/rayservice-face-recog-head-svc --address 0.0.0.0 8265:8265
```

Ray Dashboard screenshots

<img src="assets\ray_dashboard-serve-deployments.png">

<img src="assets\ray_dashboard-serve-logs.png">

<img src="assets\ray_dashboard-serve-monitoring.png">


Test `/` router

``` shell
curl 0.0.0.0:8000/
```           

Test `/uploadfile` router

``` shell
curl -X POST -H "Content-Type: multipart/form-data" -H "Authorization: Bearer TOKEN" -F "file=@your_image.jpg" localhost:8000/uploadfile
```           

Load test with locust

``` shell
pip install locust
```           

``` shell
DATASET_PATH="/.../images" locust -f locustfile.py \ 
    --headless -u NUM_CLIENTES -r RATE_CLIENTS_PER_SECOUND \
    --run-time EXECUTION_TIME -H 0.0.0.0:8000 --csv CSV_NAME  
```           

<h4> Cleanup </h4>

Delete the RayService

``` shell
kubectl delete -f kubernetes/face-recog-svc.yaml
```

Uninstall the KubeRay operator

``` shell
helm uninstall kuberay-operator
kubectl delete crd rayclusters.ray.io
kubectl delete crd rayjobs.ray.io
kubectl delete crd rayservices.ray.io
```

Uninstall the Qdrant

``` shell
helm uninstall qdrant
```

<h5> Local </h5>

You can perform local tests using Ray. Check and set necessary envs

<h6> Install Ray </h6>

``` shell
pip install ray[data,serve]==2.7.1
```

<h6> Job </h6>

Install necessary libs (check in dockerfile)

``` shell
mv jobs/job_hf.py .
python job_hf.py
```

<h6> Build new Serve Config Files </h6>

Install necessary deps

``` shell
pip install -r requirements-svc-local.txt
```

Serve Config Files define Ray Serve app

``` shell
serve build app:pipeline -o config.yaml
```

<h6> Server </h6>

``` shell
serve run config.yaml
```

<h2> Customize the project </h2>

You can edit Jobs or Service to adapt to your scenario

<h3> Build new Docker images </h3>

In the project source

``` shell
docker build -t face-recog-sys:job-description-version -f dockerfiles/Dockerfile.job.your.customizes .
```

``` shell
docker build -t face-recog-sys:svc-version -f dockerfiles/Dockerfiles.service . 
```

<h3> RayService </h3>

Use Serve Config File into Kubernetes RayService to define application

<h2> Disclaimer </h2>

Many steps were based heavily on Ray documentation

<h2> Next Features </h2>

* Add Face Segmentation using U2net-p

* Add support to TensorRT 

* Add support to OpenTelemetry