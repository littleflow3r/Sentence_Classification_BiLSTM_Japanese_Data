import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, emb_dim, out_dim, batch_size, dropout, gpu=True, vec=None):
        super().__init__()
        self.gpu = gpu
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, emb_dim)
        if vec is not None:
            self.embed.weight.data.copy_(vec)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim*2, out_dim)
    
    def init_hidden(self):
        h0 = Variable(torch.zeros(1*2, self.batch_size, self.hidden_dim))
        c0 = Variable(torch.zeros(1*2, self.batch_size, self.hidden_dim))
        if self.gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)
    
    def forward(self, seq):
        self.hidden = self.init_hidden()
        emb = self.embed(seq)
        out, hidden = self.encoder(emb, self.hidden)
        #last = (hidden[0, :, :] + hidden[1, :, :])
        y = self.hidden2label(out[-1])
        log_probs = F.log_softmax(y, dim=2)
        # last = out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]
        return log_probs
