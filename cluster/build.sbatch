#!/bin/bash
#SBATCH --account=infra01
#SBATCH -p normal
#SBATCH --time=01:00:00
#SBATCH --job-name=build-feature-probes
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

TAG="feature-probes:base"
TAG_FILE_NAME="feature-probes+25.06.sqsh"
CE_IMAGE_DIR=${SCRATCH}/ce-images/

set -euo pipefail

cd $SCRATCH/feature-probes/cluster
podman build --no-cache -t $TAG .

mkdir -p ${CE_IMAGE_DIR}
cd ${CE_IMAGE_DIR}

enroot import -o $TAG_FILE_NAME podman://$TAG
