import sys
import os

# Robustly add project root to path
# Script is in llm-lab/scripts/eval_hops.py
# Root is llm-lab/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Debug: Added {project_root} to sys.path")
print(f"Debug: CWD is {os.getcwd()}")

import torch
import torch.nn.functional as F
from llm_lab.model import ReasoningRopeGPT
from llm_lab.config import GPTConfig
from llm_lab.data.reasoning_dataset import TransitiveReasoningDataset
import yaml

def evaluate(model, hop, num_batches=20, device='cuda'):
    # Increase num_chains to have enough diverse data for large batches
    # For large hops, the number of permutations is huge, so we sample.
    ds = TransitiveReasoningDataset(num_chains=1000, chain_length=hop, is_test=True)
    
    batch_size = 32
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        num_evaluated_batches = 0
        for i in range(0, len(ds), batch_size):
            if num_evaluated_batches >= num_batches:
                break
            
            batch_indices = range(i, min(i + batch_size, len(ds)))
            if not batch_indices: break
            
            X_list = []
            Y_list = []
            
            for idx in batch_indices:
                x, y = ds[idx]
                X_list.append(x)
                Y_list.append(y)
                
            X = torch.stack(X_list).to(device)
            Y = torch.stack(Y_list).to(device)
            
            logits, _ = model(X)
            
            target_mask = (Y != -100)
            if target_mask.sum() == 0: continue
            
            batch_indices_tensor, time_indices = target_mask.nonzero(as_tuple=True)
            relevant_logits = logits[batch_indices_tensor, time_indices, :]
            relevant_targets = Y[batch_indices_tensor, time_indices]
            
            preds = relevant_logits.argmax(dim=-1)
            correct += (preds == relevant_targets).sum().item()
            total += len(relevant_targets)
            num_evaluated_batches += 1
            
    return correct / total if total > 0 else 0.0

def main():
    config_path = 'configs/reasoning_rope_deep.yaml'
    # Use absolute path or relative to project root
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), '..', config_path)
    
    model_path = 'runs/production_v6_rope_deep/model.pt'
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), '..', model_path)
    
    print(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    print(f"Loading checkpoint from {model_path}")
    # Explicitly set weights_only=False because the checkpoint contains a pickled GPTConfig class
    checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)

    if 'config' in checkpoint:
        print("Using configuration from checkpoint")
        gpt_config = checkpoint['config']
    else:
        # Initial config fallback
        model_args = dict(
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_embd=config['n_embd'],
            block_size=config['block_size'],
            bias=config['bias'],
            vocab_size=config['vocab_size'],
            dropout=config['dropout']
        )
        gpt_config = GPTConfig(**model_args)

    print(f"Initializing ReasoningRopeGPT with vocab_size={gpt_config.vocab_size}...")
    model = ReasoningRopeGPT(gpt_config)
    
    # Checkpoint is a dict with keys 'model', 'optimizer', etc.
    # Checkpoint is a dict with keys 'model', 'optimizer', etc.
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint # Fallback if it was saved directly
        
    # The keys in state_dict might have '_orig_mod.' prefix if compiled?
    # Strip it if necessary (though train.py usually handles it before saving)
    unwanted_prefix = '_orig_mod.'
    state_dict = {k.replace(unwanted_prefix, ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to('cuda')
    
    print("\nStarting Evaluation...")
    hops = [2, 3, 4, 5, 6] # Also include 2 and 3 for sanity check
    for hop in hops:
        print(f"Evaluating {hop}-hop chains...", end=' ', flush=True)
        try:
            acc = evaluate(model, hop, device='cuda')
            print(f"Accuracy: {acc*100:.2f}%")
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
