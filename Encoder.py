import torch
import torch.nn as nn
from TranformerBlock import TranformerBlock


class Encoder(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 embed_size, 
                 num_layer, 
                 heads, 
                 forward_expansion, 
                 dropout, 
                 max_len,
                 device) -> None:
        super(Encoder,self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size,embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList([
            TranformerBlock(
                embed_size,
                heads,
                dropout,
                forward_expansion
            )
            for _ in range(num_layer)
        ])

        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        N,seq_lenght = x.shape
        positions = torch.arange(0,seq_lenght).expand(N, seq_lenght).to(self.device)
        out = self.word_embedding(x) + self.position_embedding(positions)

        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out