import torch
import torch.nn as nn
from TranformerBlock import TranformerBlock
from SelfAttention import SelfAttention



class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device) -> None:
        super(DecoderBlock,self).__init__()
        self.device = device

        self.attention = SelfAttention(embed_size,heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.transform_block = TranformerBlock(embed_size, heads, dropout, forward_expansion)

        self.droput = nn.Dropout(dropout)
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.droput(self.norm1(attention + x))
        out = self.transform_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, 
                 trg_vocab_size,
                 embed_size,
                 num_layer,
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_len) -> None:
        super(Decoder,self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size,embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layer)
        ])
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_lenght = x.shape
        positions = torch.arange(0,seq_lenght).expand(N, seq_lenght).to(self.device)
        
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions)) 
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

