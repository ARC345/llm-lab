from dataclasses import dataclass
from typing import Optional

@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better.
    
    # Training params
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 5000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Architecture params
    activation: str = 'gelu'
    norm_type: str = 'layer' # layer, rms
    pos_emb: str = 'absolute' # absolute, rope
    
    # System
    device: str = 'cuda'
    compile: bool = False
