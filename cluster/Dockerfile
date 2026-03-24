FROM nvcr.io/nvidia/pytorch:25.06-py3

COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/

WORKDIR /workspace

# Copy your project
RUN git clone https://github.com/swiss-ai/feature-probes.git
WORKDIR /workspace/feature-probes

RUN git pull origin 

RUN uv pip install -e . --system --break-system-packages

ENV CUDA_CACHE_DISABLE=1


CMD ["/bin/bash"]
