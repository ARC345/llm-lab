
import os
import time
import math
import argparse
import urllib.request
import torch
from torch.nn import functional as F
from config import GPTConfig
from utils import ExperimentLogger
# Import all models
from model import GPT, ReluGPT, RmsGPT, RopeGPT, GeluGPT

# -----------------------------------------------------------------------------
# Experiment Runner
# -----------------------------------------------------------------------------

def run_experiment(experiment_cls, args):
    """
    Runs an experiment using the provided experiment_cls (subclass of GPT).
    """
    
    # Create config
    # We filter args to matching config fields
    config_dict = {k: v for k, v in vars(args).items() if k in GPTConfig.__annotations__}
    config = GPTConfig(**config_dict)
    
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Run Name Generation
    if args.comment:
        run_name = args.comment
    else:
        timestamp = int(time.time())
        run_name = f"{experiment_cls.__name__}_lr{config.learning_rate:.0e}_{timestamp}"

    # Setup Output Directory
    if args.resume_from:
         out_dir = args.resume_from
         run_name = os.path.basename(os.path.normpath(out_dir)) # Ensure consistent naming
         print(f"Resuming experiment: {run_name} from {out_dir}")
    else:
        # Sanitize run_name
        run_name = "".join(c for c in run_name if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
        out_dir = os.path.join('runs', run_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Starting new experiment: {run_name}")
        print(f"Output directory: {out_dir}")

    # Data Loading
    data_path = 'data/tinystories.txt'
    if not os.path.exists(data_path):
        os.makedirs('data', exist_ok=True)
        # Dummy generation if file doesn't exist for demo
        if not os.path.exists('data/tinystories.txt'):
             # Assume user handles download or file exists 
             pass

    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print(f"Data not found at {data_path}, using dummy text.")
        text = "Hello world " * 1000

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    config.vocab_size = vocab_size 
    print(f"Data loaded: {len(text)} chars, vocab_size={vocab_size}")

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

    # Model Instantiation
    print(f"Initializing Model: {experiment_cls.__name__}")
    model = experiment_cls(config)
    model = model.to(config.device)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")

    # Logging
    exp_logger = ExperimentLogger(
        metrics_file=os.path.join(out_dir, 'metrics.csv'),
        meta_file=os.path.join(out_dir, 'meta.jsonl')
    )
    
    # Log configuration (whether new or resumed)
    meta_payload = {'type': 'config', 'args': vars(args), 'model': experiment_cls.__name__}
    if args.resume_from:
        meta_payload['resumed'] = True
        meta_payload['resumed_from'] = args.resume_from
    
    exp_logger.log_metadata(meta_payload)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_iters)

    # Resume Logic
    start_iter = 0
    if args.resume_from:
        ckpt_path = os.path.join(out_dir, 'model.pt')
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_iter = checkpoint.get('iter', 0) + 1
            print(f"Resuming from iteration {start_iter}")
        else:
            print("Checkpoint not found, starting start_iter=0")

    # Hooks
    activation_metrics = {}
    def get_activation_hook(name):
        def hook(module, input, output):
            activation_metrics[name] = {
                'dead': (output <= 0).float().mean().item(),
                'mean': output.mean().item(),
                'std': output.std().item()
            }
        return hook

    for i, block in enumerate(model.blocks):
        # We assume standard structure for hooks or use try/except if structure varies
        # For now, assume generic Block has ffwd.net[1] as activation or just skip if fail
        try:
             # This might fail for custom blocks if structure changes, but okay for now
             block.ffwd.net[1].register_forward_hook(get_activation_hook(f'block_{i}_ffwd'))
        except:
             pass

    # Training Loop
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(20) 
            for k in range(20):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    iter_time = time.time()
    dead_perc = 0.0
    grad_norm_item = 0.0
    for iter in range(start_iter, config.max_iters):
        
        # 1. Evaluate
        if iter % config.eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | dead: {dead_perc*100:.1f}% | grad: {grad_norm_item:.2f}")
            iter_time = time.time() # Reset timer to exclude eval time from TPS
            
        # 2. Forward
        xb, yb = get_batch('train')
        activation_metrics.clear()
        
        logits, loss = model(xb, yb)
        
        # 3. Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Clip gradients and get norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        grad_norm_item = grad_norm.item()
        
        # 4. Step
        optimizer.step()
        scheduler.step()
        
        # 5. Logging
        t1 = time.time()
        dt = t1 - iter_time
        iter_time = t1
        
        tokens_per_iter = config.batch_size * config.block_size
        tps = tokens_per_iter / dt if dt > 1e-6 else 0.0
        
        if activation_metrics:
            dead_perc = sum(m['dead'] for m in activation_metrics.values()) / len(activation_metrics)
            act_mean = sum(m['mean'] for m in activation_metrics.values()) / len(activation_metrics)
            act_std = sum(m['std'] for m in activation_metrics.values()) / len(activation_metrics)
        else:
            dead_perc, act_mean, act_std = 0.0, 0.0, 0.0

        current_lr = optimizer.param_groups[0]['lr']
        
        metrics = {
            'step': iter,
            'tokens_sec': tps,
            'dead_perc': dead_perc,
            'act_mean': act_mean,
            'act_std': act_std,
            'lr': current_lr,
            'grad_norm': grad_norm.item()
        }
        
        if 'losses' in locals() and iter % 100 == 0:
            metrics['train_loss'] = losses['train'].item()
            metrics['val_loss'] = losses['val'].item()
            
        exp_logger.log_metrics(metrics)

    # Final Generation
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    gen_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
    print("\nGenerated Text:\n", gen_text)
    exp_logger.log_metadata({'type': 'generated_text', 'text': gen_text})

    # Checkpoint
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'iter': iter,
        'config': config,
    }
    torch.save(checkpoint, os.path.join(out_dir, 'model.pt'))
    print(f"Saved checkpoint to {os.path.join(out_dir, 'model.pt')}")

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment', type=str, default='', help='Comment for this run')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to experiment directory to resume from')
    parser.add_argument('--experiment', type=str, default='ReluGPT', help='Name of experiment class to run (ReluGPT, GeluGPT, RmsGPT)')
    
    # Add Config Args
    for field in GPTConfig.__annotations__:
        if not hasattr(GPTConfig, field):
            continue
        val = getattr(GPTConfig, field)
        t = type(val)
        parser.add_argument(f'--{field}', type=t, default=val)

    args = parser.parse_args()
    
    # Resolve Experiment Class
    # We can look up globals() to find the class by name
    if args.experiment in globals():
        experiment_cls = globals()[args.experiment]
    else:
        raise ValueError(f"Experiment class {args.experiment} not found in model.py")
        
    run_experiment(experiment_cls, args)