
import os
import torch
import numpy as np
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import legacy_compat

from llm_lab.utils import load_model_from_checkpoint, create_dataset

def debug_probe():
    # Load model
    checkpoint_path = 'runs/production_v2/model.pt'
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found")
        return

    print(f"Loading {checkpoint_path}...")
    model, config = load_model_from_checkpoint(checkpoint_path, device='cpu')
    
    # Load dataset
    class Args:
        dataset_type = 'reasoning'
        test_chains = 20 # Small number for debugging
    args = Args()
    
    ds = create_dataset(args, config, 'test')
    print(f"Dataset size: {len(ds)}")
    
    # Check one sample
    x, y = ds[0]
    target_pos = (y != -100).nonzero(as_tuple=True)[0]
    print(f"Sample 0 target positions: {target_pos}")
    if len(target_pos) > 0:
        print(f"Last target pos: {target_pos[-1]}")
        
    # Hook all layers
    activations = {i: [] for i in range(config.n_layer)}
    
    def get_hook(layer_i):
        def hook(module, input, output):
            # output shape (B, T, D)
            # We want [0, target_pos, :]
            # But we are inside forward, we don't know target_pos easily unless we pass it or store it.
            # Simpler: Store full output and slice later, or use single-item batch.
            activations[layer_i] = output.detach().cpu()
        return hook

    handles = []
    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(get_hook(i)))
        
    # Run one pass
    count = 0
    successful_probes = 0
    
    print("Running inference...")
    for i in range(len(ds)):
        x, y = ds[i]
        
        # Validation of target
        valid_idx = (y != -100).nonzero()
        if len(valid_idx) == 0:
            continue
            
        t_pos = valid_idx[-1].item()
        
        x = x.unsqueeze(0) # (1, T)
        
        with torch.no_grad():
            _ = model(x)
            
        # Check activations
        for layer_i in range(config.n_layer):
            act = activations[layer_i] # (1, T, D)
            # print(f"Layer {layer_i} shape: {act.shape}")
            if act.shape[1] > t_pos:
                successful_probes += 1
                
        count += 1
        if count >= 5: break
        
    print(f"Successfully probed {successful_probes} layer-samples.")
    
    # Probe Sweep
    # We will use the existing probe.py logic but loop over layers
    import subprocess
    for layer in range(config.n_layer):
        print(f"--- Probing Layer {layer} ---")
        cmd = [
            "pixi", "run", "python", "scripts/probe.py",
            "--checkpoint", checkpoint_path,
            "--layer", str(layer),
            "--output_dir", f"runs/production_v2/probing/layer_{layer}",
            "--device", "cpu"
        ]
        subprocess.run(cmd)

if __name__ == "__main__":
    debug_probe()
