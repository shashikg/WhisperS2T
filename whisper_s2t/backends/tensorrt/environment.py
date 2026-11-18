"""Locate the CUDA 13 SDK installed by NVIDIA's Python wheels."""

import os
from pathlib import Path
import sysconfig


def configure_cuda_home():
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return
    root = Path(sysconfig.get_path("purelib")) / "nvidia" / "cu13"
    if (root / "bin" / "nvcc").is_file() and (root / "include" / "cuda.h").is_file():
        os.environ["CUDA_HOME"] = str(root)
