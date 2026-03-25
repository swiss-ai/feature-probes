# Feature probes

This repo contains the implementation of feature probes, for now used for detecting entity-based hallucinations in long-form text. 

---

Please note that this is still *work in progress*, some of the scripts and approaches might not be most efficient, docs might not be complete.

For a complete list of TODOs, please refer to the [issues page](https://github.com/swiss-ai/feature-probes/issues).

## Background 

Probes can be used to detect model's behaviour by using the hidden activations. By training a small probe (usually a single layer MLP), we can predict the model's behaviour. For example we can check if it produces *hallucinations*.


## Models supported

While the code was tested for `Apertus_8B_Instruct_2509` and `Meta_Llama_3.1_8B_Instruct`, it should work for any standard language model from `transformers` library. 


## Installation

The basic installation setup for a local machine:
```

# 1. Clone and enter
git clone https://github.com/swiss-ai/feature-probes
cd feature-probes

# 2. Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create env and install (with CUDA torch on Linux)
uv sync

# 4. Store your API keys
mkdir -p ~/keys
echo "hf_..." > ~/keys/.hf_token
echo "..."    > ~/keys/.wandb_key

# 5. Run a training job
uv run python scripts/train_probe.py model=llama training=no_lora dataset=our_long_form
```
---

To set up the environment on the clariden cluster, please follow the [cluster guide](cluster/README.md).




## Acknowledgements

This repo is created by Tymoteusz Kwieciński and supervised by Anna Hedström and Imanol Schlag.
It is shared under [Apache 2.0](./LICENSE.md) license.

It was developed initailly as a project for Large Scale AI Engineering together with Klejdi Sevdari, Michał Korniak and Jack Peck.

This is the [repo](https://github.com/sevdari/hallucination_probes) of this project.

The intial project was developed as an extension of the paper [*Real-Time Detection of Hallucinated Entities in Long-Form Generation* Obeso et. al.](https://arxiv.org/abs/2509.03531).
