#!/bin/bash
# Launch DeepSpeed training on a multi-node cluster.
# Edit MODEL_PATH and DATASET_PATH to point to your model and dataset.

set -e

MODEL_PATH="/path/to/model"
DATASET_PATH="/path/to/dataset"
OUTPUT_DIR="output"

NUM_NODES=${NUM_NODES:-2}
NUM_GPUS=${NUM_GPUS:-8}
HOSTFILE=${HOSTFILE:-hostfile}

mkdir -p ${OUTPUT_DIR}

deepspeed --num_nodes ${NUM_NODES} --num_gpus ${NUM_GPUS} --hostfile ${HOSTFILE} \
  train.py \
  --deepspeed ds_config.json \
  --model_name_or_path ${MODEL_PATH} \
  --dataset_path ${DATASET_PATH} \
  --output_dir ${OUTPUT_DIR}

