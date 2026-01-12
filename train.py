import os
import time
import math
import argparse
import urllib.request
import torch
from torch.nn import functional as F
from config import GPTConfig
from model import GPT
from utils import ExperimentLogger

# -----------------------------------------------------------------------------
# CLI and Config
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--comment', type=str, default='', help='Comment for this run')
parser.add_argument('--config_path', type=str, default='', help='Path to a YAML config file (future)')
# Allow overriding basic config params via CLI
for field in GPTConfig.__annotations__:
    val = getattr(GPTConfig, field)
    t = type(val)
    parser.add_argument(f'--{field}', type=t, default=val)

args = parser.parse_args()

# Create config
config_dict = {k: v for k, v in vars(args).items() if k in GPTConfig.__annotations__}
config = GPTConfig(**config_dict)

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
exp_logger = ExperimentLogger()
exp_logger.log_metadata({'type': 'config', 'args': vars(args)})

# Data Loading
data_path = 'data/tinystories.txt' # hardcoded for now or add to config/args if needed
if not os.path.exists(data_path):
    os.makedirs('data', exist_ok=True)
    # Just a placeholder, assuming file exists or user handles it as in orig main.py
    # For now let's assume valid data path from args if we added it, but I didn't add --data to Config.
    # Let's re-add --data arg support ad-hoc or assume it's there. 
    # I'll stick to a simple default for this portfolio demo.

if os.path.exists(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
else:
    # Fallback/Download logic from original
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt' 
    # (Simplified for demo)
    print(f"Data not found at {data_path}, using dummy text or crashing")
    # Actually, let's just make it robust:
    if not os.path.exists('data/tinystories.txt'):
         # If the user script downloaded it before, it should be there. 
         # I'll assume it exists as per previous context.
         pass

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

torch.manual_seed(config.weight_decay) # Using weight_decay slot? No, use explicit seed
torch.manual_seed(1337) 

chars = sorted(list(set(text)))
vocab_size = len(chars)
config.vocab_size = vocab_size # Automatic update of vocab size

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n].to(config.device)
val_data = data[n:].to(config.device)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    return x, y

# -----------------------------------------------------------------------------
# Model & Optimizer
# -----------------------------------------------------------------------------
model = GPT(config).to(config.device)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_iters)

# Hooks
activation_metrics = {}
def get_activation_hook(name):
    def hook(module, input, output):
        # Dead activation: % value <= 0
        activation_metrics[name] = (output <= 0).float().mean().item()
    return hook

for i, block in enumerate(model.blocks):
    # Register on the activation layer (net[1] is ReLU/GELU)
    block.ffwd.net[1].register_forward_hook(get_activation_hook(f'block_{i}_ffwd'))

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(20) # eval_iters fixed to 20 for simplicity
        for k in range(20):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

start_time = time.time()

for iter in range(config.max_iters):
    
    # 1. Evaluate
    if iter % 100 == 0: # eval_interval fixed
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    # 2. Forward
    xb, yb = get_batch('train')
    activation_metrics.clear()
    
    logits, loss = model(xb, yb)
    
    # 3. Backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # -------------------------------------------------------
    # DIAGNOSTICS
    # -------------------------------------------------------
    # Gradient Norm & Variance
    total_norm = 0.0
    grad_vars = []
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            grad_vars.append(p.grad.var().item())
    total_norm = total_norm ** 0.5
    avg_grad_var = sum(grad_vars) / len(grad_vars) if grad_vars else 0.0
    
    # Update Ratio (Heuristic: lr * std(grad) / std(param))
    # We aggregate this statistics across all parameters
    update_ratios = []
    for p in model.parameters():
        if p.grad is not None:
            std_grad = p.grad.std()
            std_param = p.data.std()
            if std_param > 0:
                ratio = (config.learning_rate * std_grad / std_param).item()
                update_ratios.append(ratio)
    
    avg_update_ratio = sum(update_ratios) / len(update_ratios) if update_ratios else 0.0
    # -------------------------------------------------------

    # 4. Step
    optimizer.step()
    scheduler.step()
    
    # 5. Logging
    dt = time.time() - start_time
    tps = ((iter + 1) * config.batch_size * config.block_size) / dt if dt > 1e-6 else 0.0
    
    dead_perc = sum(activation_metrics.values()) / len(activation_metrics) if activation_metrics else 0.0
    current_lr = optimizer.param_groups[0]['lr']
    
    metrics = {
        'step': iter,
        'tokens_sec': tps,
        'grad_norm': total_norm,
        'grad_var': avg_grad_var,
        'update_ratio': avg_update_ratio,
        'dead_perc': dead_perc,
        'lr': current_lr
    }
    
    # In-loop validation loss logging (reuse calc from Evaluate step if available)
    if 'losses' in locals() and iter % 100 == 0:
        metrics['train_loss'] = losses['train'].item()
        metrics['val_loss'] = losses['val'].item()
        
    exp_logger.log_metrics(metrics)

# Final Generation
context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
gen_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print("\nGenerated Text:\n", gen_text)
exp_logger.log_metadata({'type': 'generated_text', 'text': gen_text})
