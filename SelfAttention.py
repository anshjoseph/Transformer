import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads) -> None:
        super(SelfAttention,self).__init__()
        self.embed_size= embed_size
        self.heads = heads
        self.head_dims = embed_size // heads # 256 split into 7 that cant possile so that
        assert (self.head_dims * heads == embed_size) ,"number of heads have to properly divided by the embed_size"

        self.values = nn.Linear(self.head_dims,self.head_dims, bias=False)
        self.keys = nn.Linear(self.head_dims,self.head_dims, bias=False)
        self.queries = nn.Linear(self.head_dims,self.head_dims, bias=False)
        # we doing this because we gonna send concatof values, keys and queries to garher 
        self.fc_out= nn.Linear(self.head_dims * self.heads, embed_size)
    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        # batch size

        value_len,keys_len,queries_len = values.shape[1],keys.shape[1],queries.shape[1]

        # after that we have split according to our number of heads
        values = values.reshape(N, value_len, self.heads, self.head_dims)
        keys = keys.reshape(N, keys_len, self.heads, self.head_dims)
        queries = queries.reshape(N, queries_len, self.heads, self.head_dims)

        energy = torch.einsum("nqhd,nkhd->nhqk",[queries,keys])
        # queries shape: (N, queries_len, self.heads, self.head_dims)
        # keys shape: (N, keys_len, self.heads, self.head_dims)
        # energy shape: (N, heads, queries_len, keys_len)

        if mask != None:
            energy = energy.masked_fill(mask == 0,float("-1e28"))
        """
        Attention(Q,K,V) = softmax(QK / Dk ^ 0.5)V
        """
        attention = torch.sigmoid(energy/self.embed_size**0.5, dim=3)
        out = torch.einsum("nhql,nlhd->nqhd",(attention,values))
        # Attention Shape: (N, heads, queries_len, keys_len)
        # Values Shape: (N, value_len, self.heads, self.head_dims)
        # Out Shape: (N, queries_len, heads, head_dim)        
        
        
        out = out.reshape(N, queries_len, self.head_dims * self.heads)
        # mearge each heads


        out = self.fc_out(out)
        return out
