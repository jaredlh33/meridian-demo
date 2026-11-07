# Meridian live demo backend — Render free-tier variant.
#
# Render's free web service has 512 MB of RAM, so this image runs ONE
# model only: SmolLM2-360M (Q8_0, 386 MB) with a 512-token context and
# a single inference thread. The full two-model image is demo/Dockerfile.
#
# Build downloads the GGUF from Hugging Face and verifies its SHA-256
# against the hash published in router/providers.json. The model is NOT
# copied from the build context.
#
# Deploy: connect this repo to Render as a Docker web service (free
# plan) — render.yaml in this repo already describes it — or
#   docker build -t meridian-demo-render .
#   docker run --rm -p 8181:8181 meridian-demo-render
# Health: curl localhost:8181/v1/demo/health

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# llama-cpp-python builds its C++ extension from source: needs a
# compiler + CMake. libgomp1 is the OpenMP runtime it links at run
# time. curl fetches the model file below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

# Keep the llama.cpp source build inside Render's 8 GB build limit:
# one compilation job, lighter optimization, no debug info.
ENV CMAKE_BUILD_PARALLEL_LEVEL=1
ENV CMAKE_ARGS="-DCMAKE_C_FLAGS_RELEASE=-O1 -DNDEBUG -DCMAKE_CXX_FLAGS_RELEASE=-O1 -DNDEBUG"

RUN pip install -r requirements.txt

COPY provider/ ./provider/
COPY router/ ./router/
COPY demo/ ./demo/

# Weights, fetched at build time and hash-checked against the registry.
#   SmolLM2-360M-Instruct Q8_0 <- HuggingFaceTB/SmolLM2-360M-Instruct-GGUF
ARG SMOL_SHA256=48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201
RUN mkdir -p models \
    && curl -L --fail -o models/smollm2-360m-instruct-q8_0.gguf \
        "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q8_0.gguf" \
    && echo "$SMOL_SHA256  models/smollm2-360m-instruct-q8_0.gguf" | sha256sum -c -

# Render sets $PORT (default 10000); plain docker runs use 8181.
EXPOSE 8181
CMD python demo/server.py --host 0.0.0.0 --port ${PORT:-8181} \
    --model models/smollm2-360m-instruct-q8_0.gguf --model-id smollm2-360m \
    --model2 "" --threads 1 --n-ctx 512
