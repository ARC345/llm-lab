import os
import csv
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F

class settings:
    block_size = 8
    batch_size = 4
    train_test_split = 0.9
    torch_seed = 1337
    max_iters = 5000
    eval_interval = 100
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 64
    n_head = 4
    n_layer = 4
    dropout = 0.0
        
with open('data/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
torch.manual_seed(settings.torch_seed)

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)
print("tensor->data\n  shape:", data.shape)
print("  type:", data.dtype)
    
n = int(settings.train_test_split*len(data))
train_data = data[:n]
val_data = data[n:]    

train_data[:settings.block_size+1]

x = train_data[:settings.block_size]
y = train_data[1:settings.block_size+1]

# generate a small batch of data of inputs x and targets y
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - settings.block_size, (settings.block_size,))
    x = torch.stack([data[i:i+settings.block_size] for i in ix])
    y = torch.stack([data[i+1:i+settings.block_size+1] for i in ix])
    x, y = x.to(settings.device), y.to(settings.device)
    return x, y

xb, yb = get_batch('train')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(settings.eval_iters)
        for k in range(settings.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(settings.n_embd, head_size, bias=False)
        self.query = nn.Linear(settings.n_embd, head_size, bias=False)
        self.value = nn.Linear(settings.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(settings.block_size, settings.block_size)))

        self.dropout = nn.Dropout(settings.dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(settings.n_embd, settings.n_embd)
        self.dropout = nn.Dropout(settings.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(settings.n_embd, 4 * settings.n_embd),
            nn.ReLU(),
            nn.Linear(4 * settings.n_embd, settings.n_embd),
            nn.Dropout(settings.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, settings.n_embd)
        self.position_embedding_table = nn.Embedding(settings.block_size, settings.n_embd)
        self.blocks = nn.Sequential(*[Block(settings.n_embd, n_head=settings.n_head) for _ in range(settings.n_layer)])
        self.ln_f = nn.LayerNorm(settings.n_embd) # final layer norm
        self.lm_head = nn.Linear(settings.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=settings.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -settings.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel().to(settings.device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)

# ============================================================================
# Experiment Tracking System
# ============================================================================
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Create unique run ID based on timestamp
run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join(results_dir, f'run_{run_id}')
os.makedirs(run_dir, exist_ok=True)

# Snapshot all hyperparameters
current_hparams = {
    'block_size': settings.block_size,
    'batch_size': settings.batch_size,
    'train_test_split': settings.train_test_split,
    'torch_seed': settings.torch_seed,
    'max_iters': settings.max_iters,
    'eval_interval': settings.eval_interval,
    'learning_rate': settings.learning_rate,
    'eval_iters': settings.eval_iters,
    'n_embd': settings.n_embd,
    'n_head': settings.n_head,
    'n_layer': settings.n_layer,
    'dropout': settings.dropout,
    'device': settings.device,
    'vocab_size': vocab_size,
    'run_id': run_id,
    'started_at': datetime.utcnow().isoformat(),
}

# Save hyperparameters for this run
hyperparams_path = os.path.join(run_dir, 'hyperparams.json')
try:
    with open(hyperparams_path, 'w', encoding='utf-8') as f:
        json.dump(current_hparams, f, indent=2)
    print(f"Experiment tracking initialized: run_{run_id}")
except Exception as e:
    print(f'Warning: failed to write hyperparams: {e}')

# Track best validation loss
best_val_loss = float('inf')
best_val_iter = 0

for iter in range(settings.max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % settings.eval_interval == 0 or iter == settings.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Track best validation loss
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            best_val_iter = iter

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=settings.device)
generated = decode(model.generate(context, max_new_tokens=2000)[0].tolist())
print(generated)

# Save generated output
try:
    gen_path = os.path.join(run_dir, 'generated_text.txt')
    with open(gen_path, 'w', encoding='utf-8') as f:
        f.write(f"# Generated on {datetime.utcnow().isoformat()}\n")
        f.write(f"# Run ID: {run_id}\n\n")
        f.write(generated)
except Exception as e:
    print('Warning: failed to save generated output:', e)

# ============================================================================
# Save Final Experiment Summary
# ============================================================================
final_losses = estimate_loss()
experiment_summary = {
    'run_id': run_id,
    'started_at': current_hparams['started_at'],
    'completed_at': datetime.utcnow().isoformat(),
    'hyperparameters': {k: v for k, v in current_hparams.items() if k not in ['run_id', 'started_at']},
    'final_metrics': {
        'final_train_loss': float(final_losses['train']),
        'final_val_loss': float(final_losses['val']),
        'best_val_loss': float(best_val_loss),
        'best_val_iter': best_val_iter,
        'total_iters': settings.max_iters,
        'num_params_M': round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
    },
}

# Save experiment summary for this run
summary_path = os.path.join(run_dir, 'experiment_summary.json')
try:
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_summary, f, indent=2)
except Exception as e:
    print(f'Warning: failed to write experiment summary: {e}')

# Update master summary file (all experiments)
master_summary_path = os.path.join(results_dir, 'all_experiments_summary.json')
all_experiments = []
if os.path.exists(master_summary_path):
    try:
        with open(master_summary_path, 'r', encoding='utf-8') as f:
            all_experiments = json.load(f)
    except Exception:
        all_experiments = []

# Add this experiment to the list
all_experiments.append(experiment_summary)

# Save updated master summary
try:
    with open(master_summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_experiments, f, indent=2)
    print(f"\n✓ Experiment saved: run_{run_id}")
    print(f"  Final train loss: {final_losses['train']:.4f}")
    print(f"  Final val loss: {final_losses['val']:.4f}")
    print(f"  Best val loss: {best_val_loss:.4f} (at iter {best_val_iter})")
    print(f"  Results directory: {run_dir}")
    print(f"  Master summary: {master_summary_path}")
except Exception as e:
    print(f'Warning: failed to update master summary: {e}')

# Create a human-readable comparison CSV
comparison_csv_path = os.path.join(results_dir, 'experiments_comparison.csv')
try:
    # Extract key metrics for comparison
    comparison_data = []
    for exp in all_experiments:
        row = {
            'run_id': exp['run_id'],
            'started_at': exp['started_at'],
            'learning_rate': exp['hyperparameters']['learning_rate'],
            'n_embd': exp['hyperparameters']['n_embd'],
            'n_head': exp['hyperparameters']['n_head'],
            'n_layer': exp['hyperparameters']['n_layer'],
            'dropout': exp['hyperparameters']['dropout'],
            'batch_size': exp['hyperparameters']['batch_size'],
            'block_size': exp['hyperparameters']['block_size'],
            'max_iters': exp['hyperparameters']['max_iters'],
            'final_train_loss': exp['final_metrics']['final_train_loss'],
            'final_val_loss': exp['final_metrics']['final_val_loss'],
            'best_val_loss': exp['final_metrics']['best_val_loss'],
            'best_val_iter': exp['final_metrics']['best_val_iter'],
            'num_params_M': exp['final_metrics']['num_params_M'],
        }
        comparison_data.append(row)
    
    # Write CSV
    if comparison_data:
        fieldnames = list(comparison_data[0].keys())
        with open(comparison_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_data)
        print(f"  Comparison CSV: {comparison_csv_path}")
except Exception as e:
    print(f'Warning: failed to create comparison CSV: {e}')
