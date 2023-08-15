# Importing Torch and all that good stuff
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionHead(nn.Module):
    
    def __init__(self, n_embed:int, head_size:int, dropout_rate:float=0.1):
        
        super().__init__()

        # Initialize parameters
        self.n_embed = n_embed
        self.head_size = head_size

        # Query, Key, Value
        self.W_Q = nn.Linear(n_embed, head_size)
        self.W_K = nn.Linear(n_embed, head_size)
        self.W_V = nn.Linear(n_embed, head_size)

        # Masking buffer
        self.register_buffer('mask', torch.tril(torch.ones(head_size, head_size)))

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Get the shape of the input
        _,T,C = x.shape

        # Get the query, key, and value vectors
        q = self.W_Q(x) 
        k = self.W_K(x) 
        v = self.W_V(x)

        # Masking the attention weights ala Decoder model
        wei = q @ k.transpose(-2,-1) * x.shape[-1]**-0.5 
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Matrix multiplication to get the output
        out = wei @ v

        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed:int, n_head:int, dropout_rate:float=0.1):
        super().__init__()

        # Initialize parameters
        self.head_size = n_embed // n_head
        self.n_head = n_head
        self.n_embed = n_embed
        
        # Initialize the heads
        self.heads = nn.Sequential(*[AttentionHead(n_embed, self.head_size, dropout_rate) for _ in range(n_head)])
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Projection layer
        self.W_O = nn.Linear(n_head * self.head_size, n_embed, bias=False)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        x = torch.cat([self.heads[i](x) for i in range(self.n_head)], dim=-1)
        x = self.dropout(x)
        out = self.W_O(x)

        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embed:int, dropout_rate:float=0.1):
        super().__init__()

        # Implementation of Feedforward model
        self.stack = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.stack(x)
    
class TransformerLayer(nn.Module):
    def __init__(self, n_embed:int, n_head:int, dropout_rate:float=0.1):
        super().__init__()

        # Initialize layers
        self.attention = MultiHeadAttention(n_embed, n_head, dropout_rate)
        self.feed_forward = FeedForward(n_embed, dropout_rate)
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)

    def forward(self, x:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        x = x + self.attention(self.layernorm_1(x))
        x = x + self.feed_forward(self.layernorm_2(x))
        return x
    
class TransformerModel(nn.Module):
    def __init__(self, vocab_size:int, n_embed:int, n_heads:int, block_size:int, n_layers:int=4, dropout_rate:float=0.1):
        super().__init__()

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding = nn.Embedding(block_size, n_embed)

        # Initialize the transformer layers
        self.hidden_layers = nn.Sequential(*[TransformerLayer(n_embed, n_heads, dropout_rate) for _ in range(n_layers)])

        # Final projection and normalization
        self.layernorm = nn.LayerNorm(n_embed)
        self.linear = nn.Linear(n_embed, vocab_size)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x) + self.pos_embedding(torch.arange(x.shape[1], device=x.device))
        x = self.hidden_layers(x)
        x = self.layernorm(x)
        x = self.linear(x)
        return x