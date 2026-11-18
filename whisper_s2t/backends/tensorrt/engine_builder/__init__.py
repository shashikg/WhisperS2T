"""TensorRT-LLM 1.2.1 engine conversion and build orchestration."""

import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch
from .download_utils import SAVE_DIR, download_model

TRT_LLM_VERSION = "1.2.1"


class TRTBuilderConfig:
    def __init__(
        self,
        max_batch_size=8,
        max_beam_width=1,
        max_input_len=4,
        max_output_len=444,
        dtype="float16",
        quantization=None,
        **kwargs,
    ):
        if kwargs:
            raise ValueError(f"Unsupported legacy build options: {sorted(kwargs)}")
        if dtype != "float16":
            raise ValueError("This builder currently supports float16 only")
        if quantization not in (None, "int8"):
            raise ValueError("Supported quantization modes: None, int8 (W8A16)")
        if quantization is not None:
            self.quantization = quantization
        if min(max_batch_size, max_beam_width, max_input_len, max_output_len) < 1:
            raise ValueError("Build limits must be positive")
        max_output_len = min(max_output_len, 448 - max_input_len)
        if max_output_len < 1:
            raise ValueError(
                "Decoder prompt must leave room within Whisper's 448 positions"
            )
        self.max_batch_size = max_batch_size
        self.max_beam_width = max_beam_width
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.dtype = dtype
        self.tensorrt_llm_version = importlib.metadata.version("tensorrt_llm")
        if self.tensorrt_llm_version != TRT_LLM_VERSION:
            raise RuntimeError(f"Install tensorrt-llm=={TRT_LLM_VERSION}")
        self.cuda_compute_capability = list(torch.cuda.get_device_capability())
        self.gpu_name = torch.cuda.get_device_name()
        self.tensorrt_version = importlib.metadata.version("tensorrt")
        self.torch_version = str(torch.__version__)

    def identifier(self):
        return hashlib.sha256(
            json.dumps(vars(self), sort_keys=True).encode()
        ).hexdigest()[:24]


def load_trt_build_config(output_dir):
    path = Path(output_dir)
    with (path / "trt_build_args.json").open() as f:
        config = json.load(f)
    if importlib.metadata.version("tensorrt_llm") != TRT_LLM_VERSION:
        raise ValueError("Runtime version differs; install tensorrt-llm==1.2.1")
    if config.get("tensorrt_llm_version") != TRT_LLM_VERSION:
        raise ValueError("Legacy engine: rebuild with TensorRT-LLM 1.2.1")
    if config.get("tensorrt_version") != importlib.metadata.version("tensorrt"):
        raise ValueError("TensorRT version differs; rebuild this engine")
    if config.get("torch_version") != str(torch.__version__):
        raise ValueError("PyTorch version differs; rebuild this engine")
    if config["cuda_compute_capability"] != list(torch.cuda.get_device_capability()):
        raise ValueError("Engine GPU architecture differs; rebuild on this GPU")
    if config["gpu_name"] != torch.cuda.get_device_name():
        raise ValueError("Engine GPU model differs; rebuild on this GPU")
    return SimpleNamespace(**config)


def _build_trt_engine(model_name="small", args=None, force=False, log_level="warning"):
    args = args or TRTBuilderConfig()
    output = Path(SAVE_DIR) / model_name / args.identifier()
    required = [
        "tokenizer.json",
        "trt_build_args.json",
        "encoder/config.json",
        "decoder/config.json",
        "encoder/rank0.engine",
        "decoder/rank0.engine",
    ]
    if not force and all((output / name).is_file() for name in required):
        load_trt_build_config(output)
        return str(output)
    checkpoint, tokenizer = download_model(model_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage the entire build; failures never publish an incomplete cache entry.
    with tempfile.TemporaryDirectory(prefix="build-", dir=output.parent) as temp:
        stage = Path(temp)
        (stage / "assets").mkdir()
        os.symlink(checkpoint, stage / "assets" / f"{model_name}.pt")
        conversion = [
            sys.executable,
            str(Path(__file__).with_name("convert_checkpoint.py")),
            "--model_dir",
            str(stage / "assets"),
            "--model_name",
            model_name,
            "--dtype",
            args.dtype,
            "--output_dir",
            str(stage / "weights"),
        ]
        if getattr(args, "quantization", None) == "int8":
            conversion += ["--use_weight_only", "--weight_only_precision", "int8"]
        subprocess.run(conversion, check=True)
        for component in ("encoder", "decoder"):
            command = [
                "trtllm-build",
                "--checkpoint_dir",
                str(stage / "weights" / component),
                "--output_dir",
                str(stage / component),
                "--max_batch_size",
                str(args.max_batch_size),
                "--moe_plugin",
                "disable",
                "--remove_input_padding",
                "disable",
                "--kv_cache_type",
                "continuous",
                "--log_level",
                log_level,
            ]
            if component == "encoder":
                command += [
                    "--max_input_len",
                    "3000",
                    "--max_seq_len",
                    "3000",
                    "--gemm_plugin",
                    "disable",
                    "--bert_attention_plugin",
                    args.dtype,
                ]
            else:
                command += [
                    "--max_input_len",
                    str(args.max_input_len),
                    "--max_seq_len",
                    str(args.max_input_len + args.max_output_len),
                    "--max_encoder_input_len",
                    "3000",
                    "--max_beam_width",
                    str(args.max_beam_width),
                    "--gemm_plugin",
                    args.dtype,
                    "--bert_attention_plugin",
                    args.dtype,
                    "--gpt_attention_plugin",
                    args.dtype,
                ]
            subprocess.run(command, check=True)
        shutil.copyfile(tokenizer, stage / "tokenizer.json")
        (stage / "trt_build_args.json").write_text(json.dumps(vars(args), indent=2))
        if not all((stage / name).is_file() for name in required):
            raise RuntimeError("Engine build did not produce all required artifacts")
        shutil.rmtree(stage / "weights")
        shutil.rmtree(stage / "assets")
        if output.exists():
            shutil.rmtree(output)
        stage.rename(output)
    return str(output)


def build_trt_engine(model_name="small", args=None, force=False, log_level="warning"):
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    with (Path(SAVE_DIR) / "modern-engine-build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _build_trt_engine(model_name, args, force, log_level)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
