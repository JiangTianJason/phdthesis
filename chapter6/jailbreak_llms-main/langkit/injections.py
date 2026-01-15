from copy import deepcopy
from typing import Dict, List, Optional, Union
from whylogs.core.stubs import pd
from whylogs.experimental.core.udf_schema import register_dataset_udf
from langkit import LangKitConfig, lang_config, prompt_column
from sentence_transformers import SentenceTransformer
import requests
from io import BytesIO
import numpy as np
import faiss
from langkit.utils import _get_data_home
import os
import torch

_prompt = prompt_column
_index_embeddings = None
_transformer_model = None


_USE_CUDA = torch.cuda.is_available() and not bool(
    os.environ.get("LANGKIT_NO_CUDA", False)
)
_device = "cuda" if _USE_CUDA else "cpu"


def download_embeddings(url):
    response = requests.get(url)
    data = BytesIO(response.content)
    array = np.load(data)
    return array


def init(
    transformer_name: Optional[str] = None,
    version: Optional[str] = None,
    config: Optional[LangKitConfig] = None,
):
    config = config or deepcopy(lang_config)

    global _transformer_model
    global _index_embeddings

    global _harm_embeddings_on_train        ###The embedding of "jailbreak" (in "balanced_jailbreak_dataset_train_balanced.csv") transformed by all-MiniLM-L6-v2

    if not transformer_name:
        transformer_name = "all-MiniLM-L6-v2"
    if not version:
        version = "v1"
    _transformer_model = SentenceTransformer(r"all-MiniLM-L6-v2", device=_device)       ###The model is saved locally

    path = f"index_embeddings_{transformer_name}_harm_{version}.npy"
    embeddings_url = config.injections_base_url + path
    embeddings_path = os.path.join(_get_data_home(), path)

    try:
        harm_embeddings = np.load(embeddings_path)

        _harm_embeddings_on_train = np.load(r"../../embed_result/all-MiniLM-L6-v2_on_train_harm.npy")

        save_embeddings = False
    except FileNotFoundError:
        try:
            harm_embeddings = download_embeddings(embeddings_url)

        except Exception as download_error:
            raise ValueError(
                f"Injections - unable to download embeddings from {embeddings_url}. Error: {download_error}"
            )
        save_embeddings = True
    except Exception as load_error:
        raise ValueError(
            f"Injections - unable to load embeddings from {embeddings_path}. Error: {load_error}"
        )

    try:
        _index_embeddings = faiss.deserialize_index(harm_embeddings)
        if save_embeddings:
            try:
                serialized_index = faiss.serialize_index(_index_embeddings)
                np.save(embeddings_path, serialized_index)
            except Exception as serialization_error:
                raise ValueError(
                    f"Injections - unable to serialize index to {embeddings_path}. Error: {serialization_error}"
                )
    except Exception as deserialization_error:
        raise ValueError(
            f"Injections - unable to deserialize index to {embeddings_path}. Error: {deserialization_error}"
        )


@register_dataset_udf([_prompt], f"{_prompt}.injection")
def injection(prompt: Union[Dict[str, List], pd.DataFrame]) -> Union[List, pd.Series]:
    global _transformer_model
    global _index_embeddings
    global _harm_embeddings_on_train

    if _transformer_model is None:
        raise ValueError("Injections - transformer model not initialized")
    embeddings = _transformer_model.encode(prompt[_prompt])

    ### Customed embedding on the TRAIN dataset
    target_norms = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )

    cosine_similarities = np.dot(_harm_embeddings_on_train, target_norms.T)
    max_similarities = np.max(cosine_similarities, axis=0)
    max_indices = np.argmax(cosine_similarities, axis=0)
    ############################

    faiss.normalize_L2(embeddings)
    if _index_embeddings is None:
        raise ValueError("Injections - index embeddings not initialized")
    dists, _ = _index_embeddings.search(x=embeddings, k=1)

    if max_similarities[0] >dists[0][0]:
        return np.asarray(max_similarities).flatten().tolist()
    
    return dists.flatten().tolist()


init()
