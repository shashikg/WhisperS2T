# TensorRT-LLM refresh

This development fork targets TensorRT-LLM **1.2.1** (stable), Python 3.12,
FP16 or INT8 weight-only (W8A16), and one NVIDIA GPU. NVIDIA's newer development branch removes serialized
engine builds, so upgrading to that branch requires a different runtime.

The builder converts official OpenAI checkpoints using NVIDIA's 1.2.1 converter,
then builds separate encoder and decoder engines. The static Python runtime is
adapted from NVIDIA's corresponding Whisper example. See the Apache-2.0 notices
in the adapted files. The rest of WhisperS2T retains its MIT license.

## Build and transcribe

```sh
docker build -f Dockerfile.trt-modern -t whisper-s2t:modern .
docker run --rm --gpus all --shm-size 2g \
  -v "$PWD/model-cache:/root/.cache/whisper_s2t" \
  whisper-s2t:modern python tools/build_trt.py --model small
```

Load the printed engine directory, or pass `small` to build/reuse the new cache:

```python
import whisper_s2t
model = whisper_s2t.load_model("small", backend="TensorRT-LLM")
print(model.transcribe(["sample.wav"], lang_codes=["en"], batch_size=1))
```

Use the same cache mount when running inference and mount your audio into the
container. Default engine limits: batch 8, beam 1, prompt 4 tokens, generated
output up to 444 tokens. To use initial prompts, increase `--max-input-len`;
the builder reduces the output limit to stay within 448 total positions.

Old 0.8 engines are incompatible. Do not overwrite the production ZIP with an
unvalidated engine. Cache identities include the runtime version, GPU name,
compute capability and build limits. Engines built on an RTX 4070 Ti must be
rebuilt and tested on the production L4; matching compute capability alone does
not establish engine portability. Downloads verify the official checkpoint
SHA256. Conversion/build failures propagate and do not publish incomplete
engine directories. Builds sharing this cache serialize with a file lock.

## INT8 weight-only build

```sh
python tools/build_trt.py --model small --quantization int8
```

This converts linear weights in both encoder and decoder to INT8 using
NVIDIA's W8A16 converter. Activations and unquantized layers remain FP16;
this is not full W8A8 quantization and requires no calibration dataset.
The quantization setting is part of the cache identity, so the FP16 engine
remains available separately. Load the printed INT8 engine directory through
the same `whisper_s2t.load_model(path, backend="TensorRT-LLM")` API.

## Checks

```sh
python -m unittest discover -s tests -v
python -m compileall -q whisper_s2t tools/build_trt.py
pip check
```

## Local validation (2026-11-18)

Validated with Python 3.12, PyTorch 2.9.1+cu130, CUDA runtime 13.0,
TensorRT 10.14.1.48.post1, TensorRT-LLM 1.2.1, and driver 595.99.02 on an
RTX 4070 Ti (12 GB, SM 8.9). `requirements-trt.lock` records the complete
resolved Linux/Python 3.12 environment; the Dockerfiles install this lock.
`requirements-trt.txt` retains the direct dependency constraints.

The official Small checkpoint SHA256 is
`9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794`.
Both FP16 engines built successfully (approximately 173 MiB encoder and
401 MiB decoder), and the builder subsequently reused the completed cache.
Builder unit tests, Python compilation and `pip check` passed.

All six duetx-nemo audio fixtures returned nonempty transcripts in mixed
English/Chinese/Thai batches, both with and without VAD. Observed batch times
were 1.751 s without VAD and 2.869 s with VAD. These include first-call work
and are smoke-test measurements, not a controlled benchmark. No WER/CER claim
is made; conversion preserves the pretrained Small model, including its
accuracy limitations.

To reproduce the GPU smoke check inside the container, mount the six audio
fixtures under `/audio` and an output directory under `/results`, then run:

```sh
python tools/validate_trt.py --engine /path/to/engine \
  --audio-dir /audio --output /results/validation.json
```

The duetx-nemo production dependency pin and prebuilt engine URL are unchanged.
A production L4 rollout requires a fresh L4 build, application dependency
compatibility checks, and validation on production audio. An explicit local
engine path can be selected through the application's existing
`WHISPER_MODEL_ID` setting once its runtime image uses this fork.

The validation used an isolated Docker container. The supplied Dockerfiles
capture the tested OS packages and locked Python environment; a separate clean
image build has not been run. Other WhisperS2T inference backends and quantization modes beyond W8A16
have not been validated by this refresh.


## INT8 validation (2026-11-18)

Both engine configurations report `quant_algo: W8A16`. Combined encoder and
decoder size is 375,929,848 bytes (358.5 MiB), versus approximately 574 MiB
for FP16. All six English/Chinese/Thai fixtures returned nonempty output both
with and without VAD on the RTX 4070 Ti. These are smoke checks, not WER/CER
validation. Five builder tests cover cache separation, INT8 converter arguments,
invalid settings, compatibility rejection and failed-build cleanup.
