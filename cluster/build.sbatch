#!/bin/bash
#SBATCH --account=infra01
#SBATCH -p normal
#SBATCH --time=01:00:00
#SBATCH --job-name=build-feature-probes
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err


set -euo pipefail
trap 'echo "Build failed at $(date)" >&2' ERR


# default to main, override with --export=ALL,BRANCH=refactor

BRANCH=${BRANCH:-main}

TAG="feature-probes:${BRANCH}"
TAG_FILE_NAME="feature-probes+25.06-${BRANCH}.sqsh"
CE_IMAGE_DIR=${SCRATCH}/ce-images

BRANCH=${BRANCH:-main}   

echo "Building image from branch: $BRANCH"
echo "Saving image to: ${CE_IMAGE_DIR}/${TAG_FILE_NAME}"


podman build --no-cache --build-arg BRANCH=$BRANCH -t $TAG \
    $SCRATCH/feature-probes/cluster

mkdir -p ${CE_IMAGE_DIR}

enroot import -o $TAG_FILE_NAME podman://$TAG

echo "Image built: $CE_IMAGE_DIR/$TAG_FILE_NAME"
