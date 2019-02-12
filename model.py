import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class LSTMAttn(nn.Module):
    def __init__(self, vocab_size, hidden_dim, emb_dim, out_dim, batch_size, gpu=True):
        super().__init__()
        self.gpu = gpu
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)
    
    def attn_net(self, lstm_out, final_state):
        #lstm_out = final output of lstm which contains hidden layer outputs for each sequence
        #final_state = final time-step hidden state of the lstm
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_out.transpose(1,2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def forward(self, input_sent):
        inputx = self.embed(input_sent)
        #(h0, c0) = self.init_hidden()
        #print (inputx.size())
        output, (final_h_state, final_c_state) = self.encoder(inputx)
        output = output.permute(1,0,2)
        attn_output = self.attn_net(output, final_h_state)
        logits = self.fc(attn_output)
        return logits

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, emb_dim, out_dim, batch_size, nlayers, bidir, dropout, gpu=True, vec=None):
        super().__init__()
        self.gpu = gpu
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, emb_dim)
        if vec is not None:
            self.embed.weight.data.copy_(vec)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=nlayers, bidirectional=bidir, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    #actually we probably dont need this
    def init_hidden(self):
        h0 = Variable(torch.zeros(1*2, self.batch_size, self.hidden_dim))
        c0 = Variable(torch.zeros(1*2, self.batch_size, self.hidden_dim))
        if self.gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)
    
    def forward(self, seq):
        #self.hidden = self.init_hidden()
        emb = self.dropout(self.embed(seq))
        out, (hidden, cell) = self.encoder(emb)
        #last = out[-1] #same
        last = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)) #same
        y = self.fc(last.squeeze(0))
        return y

class CNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_filters, filter_sizes, out_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, emb_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes)*n_filters, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x = [sent len, batch size]
        x = x.permute(1,0) #swap the axis, because cnn wants (batch, input)
        #x = [batch size, sent len]
        embedded = self.embedding(x) # [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1) # [batch size, 1, sent len, emb dim]
#The second dimension of the input into a nn.Conv2d layer must be the channel dimension. As text technically does not have a channel dimension, we unsqueeze our tensor to create one. This matches with our in_channels=1 in the initialization of our convolutional layers
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)
        
