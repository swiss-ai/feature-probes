# Feature probes

This repo contains the implementation of feature probes, for now used for detecting entity-based hallucinations in long-form text. 


Please note that this is still *work in progress*. The whole code probably will be refactored soon ;).

## Background 

Probes can be used to detect model's behaviour by using the hidden activations. By training a small probe (usually a single layer MLP), we can predict the model's behaviour. For example we can check if it produces *hallucinations*.


## Models supported

While the code was tested for `Apertus_8B_Instruct_2509` and `Meta_Llama_3.1_8B_Instruct`, it should work for any standard language model from `transformers` library. 

A future work is to validate the codebase for larger models, e.g. `Apertus_70B_Instruct_2509`.


## Installation

The basic installation setup:
```

# 1. Clone and enter
git clone https://github.com/swiss-ai/feature-probes
cd feature-probes

# 2. Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create env and install (with CUDA torch on Linux)
uv sync

# 5. Run a training sweep
uv run python scripts/train_probe.py --config configs/base.yaml
```

Installation on Clariden:
```
TODO...
```




