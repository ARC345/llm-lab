
import os
import time
import math
import argparse
import urllib.request
import yaml
import torch
from torch.nn import functional as F
from llm_lab.config import GPTConfig
from llm_lab.utils import ExperimentLogger, create_dataset, get_activation_metric_hook
# Import all models
from llm_lab.model import GPT, ReluGPT, RmsGPT, RopeGPT, GeluGPT, ReasoningGPT, ReasoningRopeGPT
from llm_lab.data.reasoning_dataset import TransitiveReasoningDataset

# -----------------------------------------------------------------------------
# Experiment Runner
# -----------------------------------------------------------------------------
def run_experiment(experiment_cls, args, parser=None):
    """
    Runs an experiment using the provided experiment_cls (subclass of GPT).
    """
    
    # Create config
    # If args.config is set, load from YAML
    if hasattr(args, 'config') and args.config:
        print(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Override args with yaml_config ONLY if arg is default
        for k, v in yaml_config.items():
            if parser and hasattr(args, k):
                if getattr(args, k) == parser.get_default(k):
                    setattr(args, k, v)
                else:
                    print(f"Config ignored for {k}: CLI value {getattr(args, k)} overrides config {v}")
            else:
                setattr(args, k, v)
            
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
    dataset_type = getattr(args, 'dataset_type', 'tinystories')
    
    if dataset_type == 'reasoning':
        train_ds = create_dataset(args, config, 'train')
        test_ds = create_dataset(args, config, 'test')
        ood_ds = create_dataset(args, config, 'ood')
        
        print(f"Reasoning Dataset: {len(train_ds)} train chains, {len(test_ds)} test chains, {len(ood_ds)} OOD chains")
        
        stoi = train_ds.node_to_id # Slightly hacky, but consistent across instances
        itos = {v:k for k,v in stoi.items()}
        # Add specials
        for k,v in train_ds.special_tokens.items():
            stoi[k] = v
            itos[v] = k
        
        decode = lambda l: ''.join([itos.get(i, '?') for i in l])
        
        def get_batch(split):
            ds = train_ds if split == 'train' else test_ds
            
            # Random sampling
            batch_indices = torch.randint(0, len(ds), (config.batch_size,))
            
            xs = []
            ys = []
            for idx in batch_indices:
                x, y = ds[idx.item()]
                xs.append(x)
                ys.append(y)
            
            x_batch = torch.stack(xs).to(config.device)
            y_batch = torch.stack(ys).to(config.device)
            
            # Sequence length depends on chain_length usually, but let's check input tensor shape directly
            # or use x_batch.shape[1] if we trust the dataset, but we want to assert it matches expectation.
            # In dataset, chain [A, B, C] -> "A→B.B→C.QA?→"
            # 2 hops: "A→B" (3) "." (1) "B→C" (3) "." (1) "Q" (1) "A" (1) "?" (1) "→" (1) = 12 tokens?
            # Let's inspect dataset more closely or just assert consistent shape within batch.
            # Better yet, capture the actual shape from the first item or known constant.
            # For now, let's just assert x_batch and y_batch have same shape and batch_size is correct.
            
            assert x_batch.shape[0] == config.batch_size, f"Batch size mismatch: {x_batch.shape[0]}"
            assert x_batch.shape == y_batch.shape, "X and Y shapes must match"
            
            return x_batch, y_batch

        def get_ood_batch():
            batch_indices = torch.randint(0, len(ood_ds), (config.batch_size,))
            xs = []
            ys = []
            for idx in batch_indices:
                x, y = ood_ds[idx.item()]
                xs.append(x)
                ys.append(y)
            return torch.stack(xs).to(config.device), torch.stack(ys).to(config.device)

    else:
        # Defaults to TinyStories
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
            
        def get_ood_batch():
            return get_batch('val') # Dummy fallback

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
    
    for i, block in enumerate(model.blocks):
        try:
             # This might fail for custom blocks if structure changes, but okay for now
             block.ffwd.net[1].register_forward_hook(
                 get_activation_metric_hook(activation_metrics, f'block_{i}_ffwd')
             )
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
            
        # OOD Eval
        if dataset_type == 'reasoning':
            losses_ood = torch.zeros(20)
            for k in range(20):
                X, Y = get_ood_batch()
                logits, loss = model(X, Y)
                losses_ood[k] = loss.item()
            out['ood'] = losses_ood.mean()
            
        model.train()
        return out

    @torch.no_grad()
    def compute_reasoning_metrics(model, num_batches=5):
        # Computes Accuracy and Error Analysis
        model.eval()
        correct = 0
        total = 0
        
        # Error counters
        errors = {
            'intermediate_error': 0,
            'query_error': 0,
            'random_error': 0, 
            'input_node_error': 0
        }
        
        # We need validation set iterator
        for _ in range(num_batches):
            X, Y = get_batch('val') # 2-hop
            logits, _ = model(X) # (B, T, V)
            
            # Find the target indices
            # Y has -100 everywhere except the true target
            # shape (B, T)
            # We want to gather the logits at the positions where Y != -100
            
            # Create a mask of valid targets
            target_mask = (Y != -100) # (B, T)
            
            if target_mask.sum() == 0: continue

            # We can flatten or use advanced indexing
            # But let's assume one target per sequence for simplicity in this specific dataset
            # (which is true: only final answer is trained)
            
            # Get indices: (batch_idx, time_idx)
            batch_indices, time_indices = target_mask.nonzero(as_tuple=True)
            
            # Select logits at these time steps
            # logits: (B, T, V) -> (N, V)
            relevant_logits = logits[batch_indices, time_indices, :]
            
            valid_preds = relevant_logits.argmax(dim=-1) # (N,)
            valid_targets = Y[batch_indices, time_indices] # (N,)
            
            correct += (valid_preds == valid_targets).sum().item()
            total += len(valid_targets)
            
            # Error Analysis
            # Need to decode to analyze
            # X[i] contains input chain.
            for i in range(len(valid_preds)):
                pred_token = valid_preds[i].item()
                target_token = valid_targets[i].item()
                
                # Get the original batch index to retrieve X
                orig_batch_idx = batch_indices[i].item()
                
                if pred_token != target_token:
                    # Analyze error
                    pred_char = itos.get(pred_token, '')
                    target_char = itos.get(target_token, '')
                    
                    # Reconstruct input input partial
                    input_seq = X[orig_batch_idx].tolist()
                    input_chars = [itos.get(t, '') for t in input_seq]
                    
                    # Categorize
                    # Heuristic:
                    # Get all nodes in input
                    nodes_in_input = set([c for c in input_chars if c in train_ds.nodes])
                    
                    if pred_char in nodes_in_input:
                         errors['input_node_error'] += 1
                         
                         # Check specific types if possible
                         # Query node is usually near the end of non-padded seq
                         # But easier is just checking if pred == query_node
                         # We can parse the input string to find 'Q' 'A' '?'
                         try:
                             q_idx = input_seq.index(train_ds.special_tokens['Q'])
                             query_node_token = input_seq[q_idx+1] # Node after Q
                             if pred_token == query_node_token:
                                 errors['query_error'] += 1
                         except:
                             pass
                    else:
                        errors['random_error'] += 1

        acc = correct / total if total > 0 else 0.0
        
        # Normalize errors
        total_errors = total - correct
        error_stats = {}
        if total_errors > 0:
            for k, v in errors.items():
                error_stats[k] = v / total_errors
        else:
             for k in errors: error_stats[k] = 0.0

        model.train()
        return acc, error_stats

    @torch.no_grad()
    def compute_ood_accuracy(model, num_batches=5):
        model.eval()
        correct = 0
        total = 0
        for _ in range(num_batches):
            X, Y = get_ood_batch()
            logits, _ = model(X)
            
            target_mask = (Y != -100)
            if target_mask.sum() == 0: continue
            
            batch_indices, time_indices = target_mask.nonzero(as_tuple=True)
            relevant_logits = logits[batch_indices, time_indices, :]
            
            valid_preds = relevant_logits.argmax(dim=-1)
            valid_targets = Y[batch_indices, time_indices]
            
            correct += (valid_preds == valid_targets).sum().item()
            total += len(valid_targets)
        model.train()
        return correct / total if total > 0 else 0.0

    iter_time = time.time()
    dead_perc = 0.0
    grad_norm_item = 0.0
    for iter in range(start_iter, config.max_iters):
        
        # 1. Evaluate
        if iter % config.eval_interval == 0:
            losses = estimate_loss()
            
            # Custom Metrics
            acc_val, error_stats = 0.0, {}
            acc_ood = 0.0
            
            if dataset_type == 'reasoning':
                acc_val, error_stats = compute_reasoning_metrics(model)
                acc_ood = compute_ood_accuracy(model)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, OOD loss {losses.get('ood',0):.4f}")
                print(f"  > Val Acc: {acc_val:.4f} | OOD Acc: {acc_ood:.4f}")
                print(f"  > Errors: {error_stats}")
                
                # Sample predictions
                print("  > Samples:")
                X_sample, Y_sample = get_batch('val')
                model.eval()
                with torch.no_grad():
                     logits_sample, _ = model(X_sample)
                     # preds_sample = logits_sample[:, -1, :].argmax(dim=-1) # Wrong
                
                # We iterate through samples to find their specific targets
                for i in range(min(5, len(X_sample))):
                    # Find target index
                    # Note: Y_sample[i] is 1D tensor
                    valid_idx = (Y_sample[i] != -100).nonzero()
                    if len(valid_idx) == 0: continue
                    target_pos = valid_idx[-1].item() # Take last one
                    
                    pred_token = logits_sample[i, target_pos, :].argmax().item()
                    target_token = Y_sample[i, target_pos].item()
                    
                    inp_str = decode(X_sample[i][:target_pos+1].tolist()).replace('[PAD]', '')
                    pred_str = itos.get(pred_token, '?')
                    tgt_str = itos.get(target_token, '?')
                    
                    status = "Correct" if pred_token == target_token else "Wrong"
                    print(f"    {inp_str} -> {pred_str} (Target: {tgt_str}) [{status}]")
                model.train()
                
            else:
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | dead: {dead_perc*100:.1f}% | grad: {grad_norm_item:.2f}")
            
            iter_time = time.time() # Reset timer to exclude eval time from TPS
            
        # 2. Forward
        xb, yb = get_batch('train')
        activation_metrics.clear()
        
        logits, loss = model(xb, yb)
        
        assert not torch.isnan(loss), f"Loss is NaN at step {iter}"
        
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
        
        if 'losses' in locals() and iter % config.eval_interval == 0:
            metrics['train_loss'] = losses['train'].item()
            metrics['val_loss'] = losses['val'].item()
            if 'acc_val' in locals():
                metrics['val_acc'] = acc_val
                metrics['ood_acc'] = acc_ood
                for k,v in error_stats.items():
                    metrics[k] = v
            
        exp_logger.log_metrics(metrics)

    # Final Generation
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    gen_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
    print("\nGenerated Text:\n", gen_text)
    exp_logger.log_metadata({'type': 'generated_text', 'text': gen_text})

    # Checkpoint
    # Checkpoint
    checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'model_args': model_args, # Removed: undefined, config covers this
                    'iter_num': iter, 
                    'final_val_loss': losses['val'] if 'losses' in locals() else -1.0,
                    'config': config,
                    'model_cls': experiment_cls.__name__,
                }
    torch.save(checkpoint, os.path.join(out_dir, 'model.pt'))
    print(f"Saved checkpoint to {os.path.join(out_dir, 'model.pt')}")

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comment', type=str, default='', help='Comment for this run')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to experiment directory to resume from')
    parser.add_argument('--experiment', type=str, default='ReluGPT', help='Name of experiment class to run (ReluGPT, GeluGPT, RmsGPT)')
    parser.add_argument('--dataset_type', type=str, default='tinystories', help='Type of dataset (tinystories, reasoning)')
    parser.add_argument('--num_chains', type=int, default=400, help='Number of chains for reasoning dataset')
    parser.add_argument('--test_chains', type=int, default=100, help='Number of test chains for reasoning dataset')
    
    # Add Config Args
    for field in GPTConfig.__annotations__:
        if not hasattr(GPTConfig, field):
            continue
        val = getattr(GPTConfig, field)
        parser.add_argument(f'--{field}', type=type(val), default=val)

    args = parser.parse_args()
    
    # Resolve Experiment Class
    # We can look up globals() to find the class by name
    if args.experiment in globals():
        experiment_cls = globals()[args.experiment]
    else:
        raise ValueError(f"Experiment class {args.experiment} not found in model.py")
        
    run_experiment(experiment_cls, args, parser)