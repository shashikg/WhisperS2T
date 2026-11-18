FROM python:3.12-slim-trixie
ARG SKIP_TENSORRT_LLM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libopenmpi40 openmpi-bin libsndfile1 ffmpeg git libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_HOME=/usr/local/lib/python3.12/site-packages/nvidia/cu13
ENV PATH="${CUDA_HOME}/bin:${PATH}"
WORKDIR /opt/whisper-s2t
COPY . .
RUN if [ -z "$SKIP_TENSORRT_LLM" ]; then pip install --no-cache-dir -r requirements-trt.lock; \
    else pip install --no-cache-dir -r requirements.txt; fi \
    && pip install --no-cache-dir --no-deps .
CMD ["/bin/bash"]
