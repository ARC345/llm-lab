
import torch
import argparse
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import legacy_compat

from llm_lab.utils import load_model_from_checkpoint, create_dataset
from llm_lab.data.reasoning_dataset import TransitiveReasoningDataset

def evaluate_ood(checkpoint_path, device='cpu', num_chains=1000):
    print(f"Loading model from {checkpoint_path}...")
    model, config = load_model_from_checkpoint(checkpoint_path, device)
    model.eval()
    
    print(f"Generating 3-hop OOD dataset ({num_chains} chains)...")
    # Manually create dataset with chain_length=3
    # Note: create_dataset utils might not expose chain_length easily if arguments are rigid.
    # So we instantiate directly.
    ds = TransitiveReasoningDataset(num_chains=num_chains, chain_length=3, seed=100, is_test=True)
    
    print(f"Dataset Size: {len(ds)}")
    # Validate sample
    x0, y0 = ds[0]
    print(f"Sample 0 inputs shape: {x0.shape}")
    
    # Iterate
    correct = 0
    total = 0
    
    print("Evaluating...")
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)
            
            # Forward
            logits, _ = model(x) # (1, T, vocab)
            
            # Get prediction at target position
            # y has -100 everywhere except target
            valid_mask = (y != -100)
            if valid_mask.sum() == 0:
                continue
                
            # Assume single target at the end
            target_val = y[valid_mask].item()
            pred_logits = logits[valid_mask] # (1, vocab)
            pred_val = pred_logits.argmax(dim=-1).item()
            
            if pred_val == target_val:
                correct += 1
            total += 1
            
            if i % 100 == 0:
                print(f"Progress {i}/{len(ds)}: Acc {correct/(i+1):.4f}")

    acc = correct / total
    print(f"Final 3-Hop OOD Accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    evaluate_ood(args.checkpoint, args.device)
