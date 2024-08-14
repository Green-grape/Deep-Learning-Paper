import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int, seq_len: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(seq_len, dim))  # learnable parameter

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        token_embedded = self.token_embedding(tokens)
        return token_embedded + self.position_embedding

# Encoder layer like Transformer


class CLIPLayer(nn.Module):
    def __init__(self, n_heads: int, dim: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(dim)
        self.attention = SelfAttention(n_heads, dim)
        self.layernorm_2 = nn.LayerNorm(dim)
        # Feed Forward Layer
        self.linear_1 = nn.Linear(dim, dim*4)
        self.linear_2 = nn.Linear(dim*4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        # Layer Normalization
        x = self.layernorm_1(x)
        # Self Attention
        x = self.attention(x, causal_mask=True)
        # Residual Connection
        x = x + residue
        residue = x
        # Layer Normalization
        x = self.layernorm_2(x)
        # Feed Forward Layer
        x = self.linear_1(x)
        x = x*torch.sigmoid(1.702*x)
        x = self.linear_2(x)
        return x+residue


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # vocab_size = 49408, dim = 768, seq_len = 77
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            # 12 attention heads, 768 embedding size
            CLIPLayer(12, 768) for _ in range(12)
        ])
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim))
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)  # layer에 대한 normalization
        return output
