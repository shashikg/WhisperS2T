"""GPU smoke test using duetx-nemo's English, Chinese and Thai audio fixtures."""

import argparse
import json
from pathlib import Path
import time

import torch
import whisper_s2t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = whisper_s2t.load_model(args.engine, backend="TensorRT-LLM")
    files = sorted(args.audio_dir.glob("*_warm_up_*.wav"))
    if len(files) != 6:
        raise RuntimeError("Expected six repository audio fixtures")
    languages = [p.name.split("_")[0] for p in files]
    results = []
    for method in ("transcribe", "transcribe_with_vad"):
        started = time.perf_counter()
        transcripts = getattr(model, method)(
            [str(p) for p in files], lang_codes=languages, batch_size=6
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if len(transcripts) != len(files):
            raise RuntimeError("Transcription lost input files")
        texts = [
            " ".join(s["text"] for s in segments).strip() for segments in transcripts
        ]
        if not all(texts):
            raise RuntimeError(f"Empty transcript from {method}: {texts}")
        results.append(
            {
                "method": method,
                "seconds": elapsed,
                "transcripts": dict(zip([p.name for p in files], texts)),
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "engine": args.engine,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(args.output.read_text())


if __name__ == "__main__":
    main()
