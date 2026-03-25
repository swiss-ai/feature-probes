#!/bin/bash
# Usage: bash cluster/build.sh [branch]
#   branch  — git branch to build (default: main)
#
# Must be run inside an interactive compute node session:
#   srun --account=infra01 --partition=normal --time=01:00:00 --pty bash

set -euo pipefail
trap 'echo "Build failed at $(date)" >&2' ERR

BRANCH=${1:-main}
REPO=/iopsstor/scratch/cscs/$USER/feature-probes
TAG="feature-probes:${BRANCH}"
TAG_FILE_NAME="feature-probes+25.06-${BRANCH}.sqsh"
CE_IMAGE_DIR=${SCRATCH}/ce-images

echo "Branch:  $BRANCH"
echo "Commit:  $(git -C $REPO rev-parse origin/$BRANCH)"
echo "Image:   ${CE_IMAGE_DIR}/${TAG_FILE_NAME}"

mkdir -p ${CE_IMAGE_DIR}

podman build \
    --no-cache \
    --build-arg BRANCH=$BRANCH \
    -t $TAG \
    $REPO/cluster

echo "Saving image to ${CE_IMAGE_DIR}/${TAG_FILE_NAME}..."

enroot import -o ${CE_IMAGE_DIR}/${TAG_FILE_NAME} podman://${TAG} || true

# Verify the file actually exists and is non-empty
[[ -s ${CE_IMAGE_DIR}/${TAG_FILE_NAME} ]] \
    && echo "Done: ${CE_IMAGE_DIR}/${TAG_FILE_NAME}" \
    || { echo "ERROR: sqsh file missing or empty" >&2; exit 1; }
