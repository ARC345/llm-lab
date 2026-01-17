#!/bin/bash
# Probe all layers for a given checkpoint

CHECKPOINT=$1
MODEL_NAME=$2
MAX_LAYER=$3 # Inclusive e.g. 3 for 4-layer model (0,1,2,3)

echo "Probing all layers for $MODEL_NAME ($CHECKPOINT)"

for layer in $(seq 0 $MAX_LAYER); do
    echo "----------------------------------------------------------------"
    echo "Probing Layer $layer..."
    echo "----------------------------------------------------------------"
    pixi run python scripts/probe.py \
        --checkpoint $CHECKPOINT \
        --layer $layer \
        --test_chains 2000 \
        --output_dir runs/$MODEL_NAME/probing/layer_$layer
    
    # Check if failed
    if [ $? -ne 0 ]; then
        echo "Probe failed for layer $layer"
        exit 1
    fi
done

echo "All probes completed for $MODEL_NAME"
