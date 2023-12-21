import os
import re
import zipfile

import ray
from huggingface_hub import hf_hub_download

from neural_search.qdrant import NeuralSearch
from predictors.batch.mobile_facenet_ort import MobileFaceNetORTBatchPredictor
from predictors.batch.ultra_light_ort import UltraLightORTBatchPredictor
from processing.batch.face_postp import BatchFacePostProcessing


def parse_filename(row):
    path = os.path.basename(row["path"])
    row["name"] = re.sub(r"(_\d+\.jpg)", "", path).replace("_", " ")
    return row


if __name__ == "__main__":
    # qdrant

    collection_name = os.getenv("QDRANT_COLLECTION_NAME", default="faces")

    url = os.getenv("QDRANT_URL", default="0.0.0.0")

    port = int(os.getenv("QDRANT_PORT", default="6333"))

    grpc_port = int(os.getenv("QDRANT_GRPC_PORT", default="6334"))

    prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", default="True")

    https = os.getenv("QDRANT_HTTPS", default="False")

    api_key_env = os.getenv("QDRANT_API_KEY", default="None")
    api_key = api_key_env if api_key_env != "None" else None

    if_create_collection = os.getenv("IF_CREATE_COLLECTION", default="False")
    create_collection = True if if_create_collection == "True" else False

    # models configs

    apply_resize_ul = os.getenv("APPLY_RESIZE_ULTRA_LIGHT", default="True")
    apply_resize = True if apply_resize_ul == "True" else False

    backend = os.getenv("BACKEND", default="openvino")

    batch_size_funcs = int(os.getenv("BATCH_SIZE_FUNCS", default="32"))

    batch_size_models = int(os.getenv("BATCH_SIZE_MODELS", default="32"))

    model_path_mob_facenet = os.getenv(
        "MODEL_PATH_MOB_FACENET", default="models/mobilefacenet_prep.onnx"
    )

    model_path_ultra_light = os.getenv(
        "MODEL_PATH_ULTRA_LIGHT", default="models/ultralight_RBF_320_prep_nms.onnx"
    )

    # scale configs

    num_actors_to_basic_func = int(os.getenv("NUM_ACTORS_TO_BASIC_FUNC", default="1"))

    num_actors_to_models = int(os.getenv("NUM_ACTORS_TO_MODELS", default="1"))

    # dataset configs

    dataset_repo_id = os.getenv("DATASET_REPO_ID") 
    filename = os.getenv("DATASET_FILENAME")

    local_dir = "downloads"
    dataset_dir = "dataset"

    print(f"Downlaod dataset {dataset_repo_id}/{filename}")

    hf_hub_download(
        repo_id=dataset_repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=local_dir,
    )

    with zipfile.ZipFile(os.path.join(local_dir, filename), "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    print("Unzip dataset")        

    print("Load images with Ray Data")

    ds = (
        ray.data.read_images(
            os.path.join(dataset_dir, filename.split(".")[0]),
            include_paths=True,
        )
        .map(parse_filename)
        .drop_columns("path")
    )

    print("Start map batch processing")

    ds_embeddings = (
        ds.map_batches(
            UltraLightORTBatchPredictor,
            batch_size=batch_size_models,
            batch_format="numpy",
            compute=ray.data.ActorPoolStrategy(size=num_actors_to_models),
            zero_copy_batch=True,
            fn_constructor_kwargs=dict(
                model_path=model_path_ultra_light,
                input_key="image",
                backend=backend,
                apply_resize=apply_resize,
            ),
        )
        .map_batches(
            BatchFacePostProcessing,
            batch_size=batch_size_funcs,
            batch_format="numpy",
            compute=ray.data.ActorPoolStrategy(size=num_actors_to_basic_func),
            zero_copy_batch=True,
            fn_constructor_kwargs=dict(input_key="original_image", output_key="face"),
        )
        .map_batches(
            MobileFaceNetORTBatchPredictor,
            batch_size=batch_size_models,
            batch_format="numpy",
            compute=ray.data.ActorPoolStrategy(size=num_actors_to_models),
            zero_copy_batch=True,
            fn_constructor_kwargs=dict(
                model_path=model_path_mob_facenet, input_key="face", backend=backend
            ),
        )
    )

    print("Batch map process finish")

    df_embeddings = ds_embeddings.to_pandas()

    embeddings = df_embeddings["embedding"].tolist()

    df = ds.to_pandas()

    payloads = [{"name": value} for value in df["name"]]

    print("Store in qdrant")

    ns = NeuralSearch(
        url=url,
        port=port,
        grpc_port=grpc_port,
        prefer_grpc=(prefer_grpc == "True"),
        https=(https == "True"),
        api_key=api_key,
    )

    if create_collection:
        ns.create_collection(
            collection_name=collection_name,
            embedding_dim=128,
        )

    ns.insert(
        collection_name=collection_name,
        embeddings=embeddings,
        payloads=payloads,
        batch_size=32,
    )

    print("Complete job")