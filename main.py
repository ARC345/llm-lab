import torch
import torch.nn as nn
from torch.nn import functional as F

class settings:
    block_size = 8
    batch_size = 4
    torch_seed = 1337
    
    train_frac = 0.8
    val_frac = 0.1
    test_frac = 1.0 - (train_frac + val_frac)

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
print("tensot->data\n  shape:", data.shape)
print("  type:", data.dtype)
    
# 80% train, 10% val, 10% test
train_end = int(settings.train_frac * len(data))
val_end = int((settings.train_frac + settings.val_frac) * len(data))
train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]
train_data[:settings.block_size+1]

x = train_data[:settings.block_size]
y = train_data[1:settings.block_size+1]

# generate a small batch of data of inputs x and targets y
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - settings.block_size, (settings.batch_size,))
    x = torch.stack([data[i:i+settings.block_size] for i in ix])
    y = torch.stack([data[i+1:i+settings.block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')