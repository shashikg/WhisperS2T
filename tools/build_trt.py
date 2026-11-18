"""Build a GPU-specific TensorRT-LLM engine without loading the ASR pipeline."""

import argparse
from whisper_s2t.backends.tensorrt.engine_builder import (
    TRTBuilderConfig,
    build_trt_engine,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="small")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-beam-width", type=int, default=1)
    parser.add_argument("--max-input-len", type=int, default=4)
    parser.add_argument("--max-output-len", type=int, default=444)
    parser.add_argument(
        "--quantization",
        choices=["int8"],
        help="INT8 weight-only (W8A16), FP16 activations",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = TRTBuilderConfig(
        max_batch_size=args.max_batch_size,
        max_beam_width=args.max_beam_width,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        quantization=args.quantization,
    )
    print(build_trt_engine(args.model, config, force=args.force))


if __name__ == "__main__":
    main()
