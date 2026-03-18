#!/bin/bash
#SBATCH --account=infra01
#SBATCH -p debug
#SBATCH --time=01:00:00
#SBATCH --job-name=build-uv-hallucination
#SBATCH --output=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/log/build-uv-hallucination-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/log/build-uv-hallucination-%j.err

set -euo pipefail

cd /iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden
podman build -t uv-hallucination:py-3.10 .

cd /capstor/scratch/cscs/tkwiecinski/hallucination-probes/

rm -f base.sqsh

enroot import -o base.sqsh podman://uv-hallucination:py-3.10
