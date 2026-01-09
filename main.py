import os
import csv
import json
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.nn import functional as F

from clearml import Task
import argparse

# Parse arguments first
parser = argparse.ArgumentParser()
parser.add_argument('--comment', type=str, default='', help='Comment/Description for this run')
args = parser.parse_args()

task = Task.init(project_name="gpt-from-scratch", task_name="train_run")
if args.comment:
    task.set_comment(args.comment)
else:
    raise ValueError("A comment is required for this run. Please provide one using --comment.")

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
    ix = torch.randint(len(data) - settings.block_size, (settings.batch_size,))
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

best_val_loss = float('inf')

for iter in range(settings.max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % settings.eval_interval == 0 or iter == settings.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Explicitly log to ClearML
        if task: 
            task.get_logger().report_scalar(title='Loss', series='train', value=losses['train'], iteration=iter)
            task.get_logger().report_scalar(title='Loss', series='val', value=losses['val'], iteration=iter)

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ============================================================================
# ClearML Logging
# ============================================================================
task.connect(settings) # Ensure settings are logged as configuration

# Log model parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"{n_params/1e6:.2f} M parameters")
task.set_parameter("model_parameters", n_params)

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=settings.device)
gen_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())

print("\nGenerated Text:\n", gen_text)
task.get_logger().report_text("Generated Sample", gen_text)

print(f"\nRun complete. Best Loss: {best_val_loss:.4f}")
