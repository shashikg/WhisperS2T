import os
import urllib
import hashlib
import warnings
from typing import List, Optional, Union

from tqdm import tqdm

from .... import CACHE_DIR

SAVE_DIR = f"{CACHE_DIR}/models/trt"
os.makedirs(SAVE_DIR, exist_ok=True)

_MODELS = {
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
}

_TOKENIZER = {
    "large-v2": "https://huggingface.co/Systran/faster-whisper-large-v2/raw/main/tokenizer.json",
    "large-v3": "https://huggingface.co/Systran/faster-whisper-large-v3/raw/main/tokenizer.json",
}

def download_model(name):
    
    url = _MODELS[name]
    expected_sha256 = url.split("/")[-2]
    
    download_path = os.path.join(SAVE_DIR, name)
    os.makedirs(download_path, exist_ok=True)
    
    model_ckpt_path = os.path.join(download_path, "pt_ckpt.pt")
    tokenizer_path = os.path.join(download_path, "tokenizer.json")
    
    if not os.path.exists(tokenizer_path):
        with urllib.request.urlopen(_TOKENIZER[name]) as source, open(tokenizer_path, "wb") as output:
            with tqdm(
                total=int(source.info().get("Content-Length")),
                ncols=80,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    pbar.update(len(buffer))

    if os.path.exists(model_ckpt_path) and not os.path.isfile(model_ckpt_path):
        raise RuntimeError(f"{model_ckpt_path} exists and is not a regular file")

    if os.path.isfile(model_ckpt_path):
        with open(model_ckpt_path, "rb") as f:
            model_bytes = f.read()
            
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return model_ckpt_path, tokenizer_path
        else:
            warnings.warn(
                f"{model_ckpt_path} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(model_ckpt_path, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                pbar.update(len(buffer))

    with open(model_ckpt_path, "rb") as f:
        model_bytes = f.read()
        if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
            raise RuntimeError(
                "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
            )

    return model_ckpt_path, tokenizer_path