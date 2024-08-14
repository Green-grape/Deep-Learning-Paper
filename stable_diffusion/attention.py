import torch
from torch import nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads

        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

    def forward(self, x, causal_mask=False):
        # x: (Batch_Size, Seq_Length, Dim)
        batch_size, seq_length, dim = x.shape
        inter_shape = (batch_size, seq_length, self.n_heads, self.d_heads)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # q, k, v: (Batch_Size, Seq_Length, Dim) -> (Batch_Size, Seq_Length, N_Heads, D_Heads) -> (Batch_Size, N_Heads, Seq_Length, D_Heads)
        q = q.view(inter_shape).transpose(1, 2)
        k = k.view(inter_shape).transpose(1, 2)
        v = v.view(inter_shape).transpose(1, 2)

        # (Batch_size, N_Heads, Seq_Length, D_Heads) @ (Batch_size, N_Heads, D_Heads, Seq_Length) -> (Batch_size, N_Heads, Seq_Length, Seq_Length)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight = weight / math.sqrt(self.d_heads)
        weight = F.softmax(weight, dim=-1)
        # (Batch_Size, N_Heads, Seq_Length, Seq_Length) @ (Batch_Size, N_Heads, Seq_Length, D_Heads) -> (Batch_Size, N_Heads, Seq_Length, D_Heads)
        output = weight @ v
        # (Batch_Size, N_Heads, Seq_Length, D_Heads) -> (Batch_Size, Seq_Length, N_Heads, D_Heads) -> (Batch_Size, Seq_Length, Dim)
        output = output.transpose(1, 2).reshape(batch_size, seq_length, dim)
        output = self.out_proj(output)
        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed//n_heads

    def forward(self, latent, context):
        # latent: (Batch_Size, Seq_Length_q, Dim_q)
        # context: (Batch_Size, Seq_length_kv, Dim_kv)=(Batch_Size, 77, 768)
        batch_size, sequence_length, d_embed = input_shape = latent.shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # (Batch_Size, Seq_Length_q, Dim_q) -> (Batch_Size, Seq_Length_q, N_Heads, D_Head) -> (Batch_Size, N_Heads, Seq_Length_q, D_Head)
        q = self.q_proj(latent).view(interim_shape).transpose(1, 2)
        k = self.k_proj(context).view(interim_shape).transpose(1, 2)
        v = self.v_proj(context).view(interim_shape).transpose(1, 2)

        w = q @ k.transpose(-1, -2)/math.sqrt(self.d_head)
        # pixel은 이후 값을 봐도 상관이 없으므로 따로 mask가 없음
        w = F.softmax(w, dim=-1)
        output = w @ v
        # (Batch_Size, N_Heads, Seq_Length_q, D_Head) -> (Batch_Size, Seq_Length_q, N_Heads, D_Head) -> (Batch_Size, Seq_Length_q, Dim_q)
        output = output.transpose(1, 2).contiguous().view(
            input_shape)

        output = self.out_proj(output)
        return output
