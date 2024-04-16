import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters setting
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
# -----------------------

with open('../data/train.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Building vocab and mapping
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])
# ----------------------------------------------

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

# DataLoading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) # stack increases the dimensions of the tensor
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# we don't want the gradients 
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input --> (batch, time-step, channels)
        B, T, C = x.shape
        k = self.key(x) #----> (B, T, hs)
        q = self.query(x) # ---> (B, T, hs)
        # affinities (attention scores)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 #---> (B, T, hs) @ (B, hs, T) --> (B, T, T)
        wei = wei.masked_fill(self.tr[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # weighted aggregate of values
        v = self.value(x) #(B, T, hs)
        out = wei @ v #(B, T, hs)
        return out
    
