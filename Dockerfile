ARG BASE_IMAGE=nvidia/cuda
ARG BASE_TAG=12.1.0-devel-ubuntu22.04

FROM ${BASE_IMAGE}:${BASE_TAG}
ARG WHISPER_S2T_VER=main
ARG SKIP_TENSORRT_LLM

WORKDIR /workspace
ENTRYPOINT []
SHELL ["/bin/bash", "-c"]

COPY ./install_tensorrt.sh install_tensorrt.sh

RUN apt update && apt-get install -y python3.10 python3-pip libsndfile1 ffmpeg git && \
    pip3 install --no-cache-dir notebook jupyterlab ipywidgets && \
    pip3 install --no-cache-dir git+https://github.com/shashikg/WhisperS2T.git@${WHISPER_S2T_VER} && \
    CUDNN_PATH=$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))') && \
    echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:'${CUDNN_PATH} >> ~/.bashrc

RUN if [[ -z "$SKIP_TENSORRT_LLM" ]]; then /bin/bash install_tensorrt.sh; fi

RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -r install_tensorrt.sh

CMD ["/bin/bash"]