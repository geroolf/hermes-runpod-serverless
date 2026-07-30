ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS llama-builder

ARG DEBIAN_FRONTEND=noninteractive
ARG LLAMA_COMMIT=3018a11e79e489b657dbb77c95694889ccff92df

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        g++ \
        git \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone https://github.com/ggml-org/llama.cpp.git \
    && cd llama.cpp \
    && git checkout "${LLAMA_COMMIT}" \
    && cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="86-real;89-real" \
        -DGGML_NATIVE=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_CURL=OFF \
    && cmake --build build --config Release --target llama-server -j"$(nproc)" \
    && strip build/bin/llama-server

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        python3 \
        python3-pip \
    && python3 -m pip install --no-cache-dir --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY --from=llama-builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY app.py /app/app.py

ENV PYTHONUNBUFFERED=1 \
    PORT=80 \
    PORT_HEALTH=80 \
    HEALTH_CHECK_PATH=/ping \
    RUNPOD_INIT_TIMEOUT=800 \
    LLAMA_HOST=127.0.0.1 \
    LLAMA_PORT=8080 \
    MODEL_REPO=oolfBER/Qwen3.6-27B-UD-Q4_K_XL-single-GGUF \
    MODEL_FILENAME=Qwen3.6-27B-UD-Q4_K_XL.gguf \
    MODEL_ALIAS=qwen3.6-27b-local

EXPOSE 80
STOPSIGNAL SIGTERM

CMD ["python3", "-u", "/app/app.py"]
