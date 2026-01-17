#!/bin/bash
set -e

# Experiment Name
RUN_NAME="production_v5_rope_10k"
CONFIG="configs/reasoning_bottleneck.yaml"

echo "Starting Training: $RUN_NAME"
# Run training using pixi environment python
# Assumes we are inside the pixi shell or using pixi run
python train.py --config $CONFIG --experiment ReasoningRopeGPT --comment $RUN_NAME

echo "Training Complete. Starting Probes..."

CHECKPOINT="runs/$RUN_NAME/model.pt"

# Layer 0
echo "Probing Layer 0..."
python scripts/probe.py --checkpoint $CHECKPOINT --layer 0 --output_dir "runs/$RUN_NAME/probing/layer_0" --device cuda

# Layer 1
echo "Probing Layer 1..."
python scripts/probe.py --checkpoint $CHECKPOINT --layer 1 --output_dir "runs/$RUN_NAME/probing/layer_1" --device cuda

# Layer 2
echo "Probing Layer 2..."
python scripts/probe.py --checkpoint $CHECKPOINT --layer 2 --output_dir "runs/$RUN_NAME/probing/layer_2" --device cuda

echo "All Done!"
