from Tranformer import Tranformer
import torch


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    x = torch.tensor([[1,5,6,4,3,9,5,2,8],[1,8,7,3,4,7,6,2,0]]).to(device)
    trg = torch.tensor([[1,7,4,3,5,9,2,8],[1,5,6,2,4,7,6,2]]).to(device)

    src_pad_index = 0
    trg_pad_index = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Tranformer(src_vocab_size,trg_vocab_size,src_pad_index,trg_pad_index).to(device)
    out = model(x, trg[:,:-1])
    print(out)
