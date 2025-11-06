# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

# --- Install dependencies ---
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        libopencv-dev \
        libprotobuf-dev protobuf-compiler \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

# --- Download and install ONNX Runtime GPU (CUDA 11.8) ---
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-gpu-1.18.0.tgz && \
    tar -xzf onnxruntime-linux-x64-gpu-1.18.0.tgz && \
    mv onnxruntime-linux-x64-gpu-1.18.0 /usr/local/onnxruntime && \
    rm onnxruntime-linux-x64-gpu-1.18.0.tgz

# --- Copy your main.cpp ---
COPY main.cpp .
ENV LD_LIBRARY_PATH=/usr/local/onnxruntime/lib:$LD_LIBRARY_PATH

# --- Compile ---
# RUN g++ main.cpp -o main \
#     -I/usr/local/onnxruntime/include \
#     -L/usr/local/onnxruntime/lib \
#     -lonnxruntime `pkg-config --cflags --libs opencv4` \
#     -ldl -lpthread -std=c++17

# # --- Set default run command ---
# CMD ["./main"]
