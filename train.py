#coding:utf-8

import torchtext
from torchtext import data, datasets

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.autograd import Variable

from model import SimpleLSTM

import random
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load('../spiece/ja.wiki.bpe.vs5000.model')

def sptokenizer(x):
    return sp.EncodeAsPieces(x)

import MeCab
tagger = MeCab.Tagger('')
tagger.parse('')

def tokenizer(text):
    wakati = []
    node = tagger.parseToNode(text).next
    while node.next:
        wakati.append(node.surface)
        node = node.next
    return wakati

TEXT = data.Field(sequential=True, tokenize=sptokenizer, lower=True)
LABEL = data.Field(sequential=False, use_vocab=False)

dataset = data.TabularDataset(path='text/sent.csv', format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
train, val, test = dataset.split(split_ratio=[0.7, 0.1, 0.2], random_state=random.getstate())
#print (len(train), len(val), len(test))
TEXT.build_vocab(train)

device = torch.device('cuda:0')
train_it, val_it, test_it = data.BucketIterator.splits((train, val, test), batch_sizes=(16,16,1), device=device, sort_key=lambda x: len(x.text), repeat=False, sort=False)

class batch_wrapper:
    def __init__(self, dl, x, y):
        self.dl, self.x, self.y = dl, x, y
    def __len__(self):
        return len(self.dl)
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x) #assuming one input
            y = getattr(batch, self.y)
            '''
            if self.y is not None: #concat the y into single tensor
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y], dim=1).float()
            else:
                y = torch.zeros((1))
            '''
            yield (X,y)


train_batch_it = batch_wrapper(train_it, 'text', 'label')
#print ('get data x and y out of batch object:', next(iter(train_batch_it)))
valid_batch_it = batch_wrapper(valid_it, 'text', 'label')
test_batch_it = batch_wrapper(test_it, 'text')

    

'''
batch = next(iter(train_it))
print (batch)
print (batch.text)
print (batch.label)
print (len(train_it))
'''
vocab_size = len(TEXT.vocab)
emb_dim = 50
hidden_dim = 50
out_dim = 2
lr = 1e-2
model = SimpleLSTM(vocab_size, hidden_dim, emb_dim, 0.2)
model.cuda()

import tqdm
opt = optim.Adam(model.parameters(), lr=lr)
loss = F.cross_entropy()
ep = 5

for ep in range(1, ep+1):
    tr_loss = 0.0
    model.train()
    for x, y in tqdm.tqdm(train_batch_it):
        opt.zero_grad()
        preds = model(x)
        loss = criterion(preds,y)
        loss.backward()
        opt.step()
        
    



