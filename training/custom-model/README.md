# Custom Model Training Template

This directory provides a minimal template for training a custom model on a
custom dataset with [DeepSpeed](https://www.deepspeed.ai/).  Only the model and
dataset paths need to be updated before launching training.

## Files

- `train.py` – simple training loop using DeepSpeed.
- `ds_config.json` – example DeepSpeed configuration.
- `run.sh` – helper script for launching multi-node training.
- `hostfile` – placeholder listing the machines in the cluster.

## Usage

1. Edit `run.sh` and update `MODEL_PATH` and `DATASET_PATH` with the desired
   model and dataset.
2. Update `hostfile` with the hostname and GPU count for each node.
3. Launch training:
   ```bash
   ./run.sh
   ```
4. Trained checkpoints will be written to the `output` directory.

`train.py` assumes the dataset contains a `text` column and will tokenize and
train a causal language model.  Adjust the script as needed for other tasks.
