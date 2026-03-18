FROM nvcr.io/nvidia/pytorch:24.10-py3

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace

# Copy your project
RUN git clone https://github.com/sevdari/hallucination_probes.git
WORKDIR /workspace/hallucination_probes

# Install your package into system Python without dependencies
# (torch from NVIDIA image stays untouched)
RUN uv pip install  . --system

# library safety tools does not really like the latest transformers
RUN uv pip install transformers==4.56.0 --system

CMD ["/bin/bash"]
