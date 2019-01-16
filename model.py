import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, emb_dim, out_dim, do):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        #self.embedding.weight.data.copy_(pretrained_vec) #load pretrained vec
        #self.embedding.weight.requires_grad = False #make embedding non-trainable    
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1, dropout=do, bidirectional=True)
        self.linear = nn.Linear(hidden_dim*2, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, seq):
        emb = self.embedding(seq)
        out, hidden  = self.encoder(emb) #initial h0, c0 is zero. _ : (hn,cn), hdn=output features for each timestep
        out = (hidden[0, :, :] + hidden[1, :, :])
        #print (out.shape)
        feature = self.linear(out)
        #print (feature.shape)
        preds = self.predictor(feature)
        #print (preds.shape)
        return preds

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, emb_dim, gpu=True, vec=None, dropout):
        super().__init__()
        self.gpu = gpu
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        if vec is not None:
            self.embed.weight.data.copy_(vec)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, dropout=dropout, bidirectional=True)
    
    def init_hidden(self, bsize):
        h0 = Variable(torch.zeros(1*2, bsize, self.hidden_dim))
        c0 = Variable(torch.zeros(1*2, bsize, self.hidden_dim))
        if self.gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)
    
    def forward(self, seq):
        self.hidden = self.init_hidden(seq)
        emb = self.embed(seq)
        out, hidden = self.encoder(emb, self.hidden)
        last = (hidden[0, :, :] + hidden[1, :, :])
        # last = out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]
        return last
