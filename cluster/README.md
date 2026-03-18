## some config files
there are some config files related to the clariden cluster, e.g. [Dockerfile](./Dockerfile) of the image, should be some sbatch scripts etc.

### build
the files (env.toml, Dockerfile, build.sbatch) are related to creating the clariden environment. 

The base `.sqsh` image with main branch cloned is saved at `/capstor/scratch/cscs/tkwiecinski/hallucination-probes/base.sqsh`. It might not have the latest repo version though. 

Notice, that there is no uv env installed in the container, there is just a global env, so don't run `uv sync`.

### running the jobs

[train.sbatch](./train.sbatch) can be used to run a sbatch job on the cluster
when it comes to setting up an interactive session, clariden docs are reaaally useful

To enable logging to huggingface, create a file `~/keys/hf_token` with your token.
