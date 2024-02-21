import torch
import torch.nn as nn
from SelfAttention import SelfAttention

class TranformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion)->None:
        super(TranformerBlock,self).__init__()
        self.attention = SelfAttention(embed_size,heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_foward = nn.Sequential(
            nn.Linear(embed_size, embed_size*forward_expansion),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size,embed_size)
        )

        self.dropout = nn.Dropout(dropout)
    def forward(self, values, keys, queries, mask):
        attention =self.attention(values,keys,mask)

        x1 = self.dropout(self.norm1(attention * queries))
        forward = self.feed_foward(x1)
        out = self.dropout(self.norm2(forward + x1))
        return out