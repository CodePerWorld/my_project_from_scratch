import torch
import torch.nn as nn

class MultiSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.head_dim = embed_size // num_heads
        assert (self.head_dim * num_heads == embed_size), "隐藏层和head数量不能整除"

    def forward(self, queries, keys, values, mask):
        N = queries.shape[0]
        queries, keys, values = self.W_q(queries), self.W_k(keys), self.W_v(keys)
        query_len, keys_len, value_len= queries.shape[1], keys.shape[1], values.shape[1]
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, keys, self.num_heads, self.head_dim)
        values = values.reshape(N, values, self.num_heads, self.head_dim)

        pre_attention = torch.einsum("nqhd,nkhd->nhkq",[queries, keys])

        if mask is not None:
            pre_attention = pre_attention.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(pre_attention / (self.head_dim ** (1/2)), dim=3)

        out = torch.einsum("nhkq,nlhd->nqhd", [attention, values]).reshape(N, queries.shape[1], self.embed_size)

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    pass