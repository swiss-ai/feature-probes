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
## Known Issues & Workarounds
To run Apertus, we need `transformers=0.46`. This is conflicted with `safetytooling` version that we use. It is best for now to install latest version of `transformers` manually, after installation.

TODO...




## License

Apache 2.0 — see LICENSE.md.


## Acknowledgements

This repo is created by Tymoteusz Kwieciński and supervised by Anna Hedstrom and Imanol Schlag.

It was developed initailly as a project for Large Scale AI Engineering together with Klejdi Sevdari, Michał Korniak and Jack Peck.
This is the [repo](https://github.com/sevdari/hallucination_probes) of this project.

The intial project was developed as an extension of the paper [*Real-Time Detection of Hallucinated Entities in Long-Form Generation* Obeso et. al.](https://arxiv.org/abs/2509.03531).