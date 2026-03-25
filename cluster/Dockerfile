FROM nvcr.io/nvidia/pytorch:25.06-py3

COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/

ARG BRANCH=main

WORKDIR /workspace

RUN git clone --branch ${BRANCH} --single-branch https://github.com/swiss-ai/feature-probes.git
WORKDIR /workspace/feature-probes


RUN uv pip install -e . --system --break-system-packages

ENV CUDA_CACHE_DISABLE=1

CMD ["/bin/bash"]
