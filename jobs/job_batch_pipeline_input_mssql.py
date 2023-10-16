import os

import ray

from db.mssql_connector import create_connection
from neural_search.qdrant import NeuralSearch
from predictors.batch.mobile_facenet_ort import MobileFaceNetORTBatchPredictor
from predictors.batch.ultra_light_ort import UltraLightORTBatchPredictor
from processing.batch.face_postp import BatchFacePostProcessing
from processing.batch.utils_batch import (
    batch_convert_bytes_to_base64,
    batch_convert_bytes_to_numpy,
    batch_convert_numpy_to_base64,
)

if __name__ == "__main__":
    # getenvs

    # mssql

    db_query = os.getenv("MSSQL_DB_QUERY")

    # qdrant

    collection_name = os.getenv("QDRANT_COLLECTION_NAME", default="faces")

    url = os.getenv("QDRANT_URL", default="0.0.0.0")

    port = int(os.getenv("QDRANT_PORT", default="6333"))

    grpc_port = int(os.getenv("QDRANT_GRPC_PORT", default="6334"))

    prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", default="True")

    https = os.getenv("QDRANT_HTTPS", default="False")

    # models configs

    apply_resize_ul = os.getenv("APPLY_RESIZE_ULTRA_LIGHT", default="True")
    apply_resize = False if apply_resize_ul == "True" else True

    backend = os.getenv("BACKEND", default="openvino")

    batch_size_funcs = int(os.getenv("BATCH_SIZE_FUNCS", default="32"))

    batch_size_models = int(os.getenv("BATCH_SIZE_MODELS", default="32"))

    model_path_mob_facenet = os.getenv(
        "MODEL_PATH_MOB_FACENET", default="models/mobilefacenet_prep.onnx"
    )

    model_path_ultra_light = os.getenv(
        "MODEL_PATH_ULTRA_LIGHT", default="models/mobilefacenet_prep.onnx"
    )

    # scale configs

    num_cpus_to_basic_func = float(os.getenv("NUM_CPUS_TO_BASIC_FUNC", default="0.3"))

    num_cpus_to_models = float(os.getenv("NUM_CPUS_TO_MODELS", default="1.0"))

    num_gpus_to_models = float(os.getenv("NUM_GPUS_TO_MODELS", default="0.0"))

    num_actors_to_basic_func = int(os.getenv("NUM_ACTORS_TO_BASIC_FUNC", default="1"))

    num_actors_to_models = int(os.getenv("NUM_ACTORS_TO_MODELS", default="1"))

    # batch app

    ds = ray.data.read_sql(db_query, create_connection)

    ns = NeuralSearch(
        url=url,
        port=port,
        grpc_port=grpc_port,
        prefer_grpc=(prefer_grpc == "True"),
        https=(https == "True"),
    )

    ns.create_collection(
        collection_name=collection_name,
        embedding_dim=128,
    )

    ds_payloads = ds.map_batches(
        batch_convert_bytes_to_numpy,
        batch_format="pandas",
        batch_size=batch_size_funcs,
        num_cpus=num_cpus_to_basic_func,
        compute=ray.data.ActorPoolStrategy(size=num_actors_to_basic_func),
        zero_copy_batch=True,
        fn_kwargs=dict(input_key="foto", output_key="image"),
    ).map_batches(
        batch_convert_bytes_to_base64,
        batch_format="pandas",
        num_cpus=num_cpus_to_basic_func,
        compute=ray.data.ActorPoolStrategy(size=num_actors_to_basic_func),
        batch_size=batch_size_funcs,
        zero_copy_batch=True,
        fn_kwargs=dict(input_key="foto", output_key="foto_base64"),
    )

    ds_embeddings = (
        ds_payloads.map_batches(
            UltraLightORTBatchPredictor,
            batch_size=batch_size_models,
            batch_format="numpy",
            num_gpus=num_gpus_to_models,
            num_cpus=num_cpus_to_models,
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
            num_cpus=num_cpus_to_basic_func,
            compute=ray.data.ActorPoolStrategy(size=num_actors_to_basic_func),
            zero_copy_batch=True,
            fn_constructor_kwargs=dict(input_key="original_image", output_key="face"),
        )
        .map_batches(
            MobileFaceNetORTBatchPredictor,
            batch_size=batch_size_models,
            batch_format="numpy",
            num_gpus=num_gpus_to_models,
            num_cpus=num_cpus_to_models,
            compute=ray.data.ActorPoolStrategy(size=num_actors_to_models),
            zero_copy_batch=True,
            fn_constructor_kwargs=dict(
                model_path=model_path_mob_facenet, input_key="face", backend=backend
            ),
        )
        .map_batches(
            batch_convert_numpy_to_base64,
            batch_format="pandas",
            batch_size=batch_size_funcs,
            zero_copy_batch=True,
            fn_kwargs=dict(input_key="face", output_key="face_base64"),
        )
    ).materialize()

    ds_payloads = ds_payloads.drop_columns(["image", "foto"])

    payloads = (
        ds_payloads.zip(ds_embeddings.select_columns(["face_base64"]))
        .to_pandas()
        .to_dict("records")
    )

    df_embeddings = ds_embeddings.to_pandas()

    embeddings = df_embeddings["embedding"].tolist()

    ns.insert(
        collection_name=collection_name,
        embeddings=embeddings,
        payloads=payloads,
        batch_size=32,
    )
