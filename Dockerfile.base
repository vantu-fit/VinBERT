FROM nvidia/cuda:12.5.1-devel-ubuntu22.04

ARG CONTAINER_TIMEZONE=Asia/Ho_Chi_Minh
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && \
    echo $CONTAINER_TIMEZONE > /etc/timezone

RUN apt update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y --no-install-recommends python3.12 python3-pip git python3-dev ninja-build && \
    ln -sf python3 /usr/bin/python && \
    ln -sf pip3 /usr/bin/pip && \
    pip install --upgrade pip

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV FORCE_CUDA="1"

RUN pip install wheel packaging && \
    pip install torch huggingface transformers Pillow numpy einops tqdm dataclasses timm matplotlib torchvision --no-cache-dir

RUN pip install flash-attn

RUN python -c "import flash_attn"

ENV SAGEMAKER_PROGRAM train.py
