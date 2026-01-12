import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import GPTConfig

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMSNorm: x * rsqrt(mean(x^2) + eps)
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, x, seq_len=None):
        # x: [batch_size, seq_len, dim]
        if seq_len is None:
            seq_len = x.shape[1]
        return self.cos[:seq_len, :], self.sin[:seq_len, :]

def apply_rope(x, cos, sin):
    # x: [batch_size, seq_len, head_dim]
    head_dim = x.shape[-1]
    x_rot = torch.cat([-x[..., head_dim//2:], x[..., :head_dim//2]], dim=-1)
    return (x * cos) + (x_rot * sin)

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        head_size = config.n_embd // config.n_head
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, rot_emb=None):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,HeadSize)
        q = self.query(x) # (B,T,HeadSize)
        v = self.value(x) # (B,T,HeadSize)
        
        if rot_emb is not None:
             cos, sin = rot_emb
             # q,k are (B,T,HeadSize). cos,sin are (T, Dim).
             # We need to reshape for broadcasting (1, seq_len, dim)
             cos = cos.unsqueeze(0)
             sin = sin.unsqueeze(0)
             q = apply_rope(q, cos, sin)
             k = apply_rope(k, cos, sin)

        # optimized attention (Flash Attention)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.config.dropout if self.training else 0, is_causal=True)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, rot_emb=None):
        out = torch.cat([h(x, rot_emb=rot_emb) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU() if config.activation == 'relu' else nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedFoward(config)
        if config.norm_type == 'rms':
            self.ln1 = RMSNorm(config.n_embd)
            self.ln2 = RMSNorm(config.n_embd)
        else:
            self.ln1 = nn.LayerNorm(config.n_embd)
            self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x, rot_emb=None):
        x = x + self.sa(self.ln1(x), rot_emb=rot_emb)
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        
        if config.pos_emb == 'absolute':
             self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        else:
             self.position_embedding_table = None
             head_size = config.n_embd // config.n_head
             self.rope = RotaryPositionalEmbedding(head_size, max_seq_len=config.block_size)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        if config.norm_type == 'rms':
            self.ln_f = RMSNorm(config.n_embd)
        else:
            self.ln_f = nn.LayerNorm(config.n_embd)
            
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        x = tok_emb
        
        rot_emb = None
        if self.config.pos_emb == 'absolute':
            pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
            x = x + pos_emb # (B,T,C)
        else:
            # RoPE
            rot_emb = self.rope(x, seq_len=T)

        for block in self.blocks:
            x = block(x, rot_emb=rot_emb) # (B,T,C)
            
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
            idx_cond = idx[:, -self.config.block_size:]
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
