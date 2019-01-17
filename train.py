#coding:utf-8

import torchtext
from torchtext import data, datasets

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.autograd import Variable

from model import BiLSTM

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
LABEL = data.Field(sequential=False)

dataset = data.TabularDataset(path='text/sent.csv', format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
train, val, test = dataset.split(split_ratio=[0.7, 0.1, 0.2], random_state=random.getstate())
#print (len(train), len(val), len(test))
TEXT.build_vocab(train, val, test)
LABEL.build_vocab(train, val, test)

bsize = 8
gpu = False
device = 'cpu'
if gpu and torch.cuda.is_available:
    device = torch.device('cuda:1')

train_it, valid_it, test_it = data.BucketIterator.splits((train, val, test), batch_sizes=(bsize,bsize,bsize), device=device, sort_key=lambda x: len(x.text), repeat=False)

'''
class batch_wrapper:
    def __init__(self, dl, x, y):
        self.dl, self.x, self.y = dl, x, y
    def __len__(self):
        return len(self.dl)
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x) #assuming one input
            y = getattr(batch, self.y)
            yield (X,y)


train_batch_it = batch_wrapper(train_it, 'text', 'label')
#print ('get data x and y out of batch object:', next(iter(train_batch_it)))
valid_batch_it = batch_wrapper(valid_it, 'text', 'label')

test_batch_it = batch_wrapper(test_it, 'text')

batch = next(iter(train_it))
print (batch)
print (batch.text)
print (batch.label)
print (len(train_it))
'''

def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)

def train(model, train_it, lossf, optimizer):
    model.train()
    avg_loss = 0.0
    truth_res, pred_res = [], []
    count = 0
    for batch in train_it:
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        #model.hidden = model.init_hidden()
        pred = model(sent)
        pred_label = pred.data.cpu().max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        model.zero_grad()
        #print ('PRED:', pred)
        #print ('LABEL:', label)
        
        loss = lossf(pred, label)
        avg_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_it)
    acc = get_accuracy(truth_res, pred_res)
    return avg_loss, acc
    

vocab_size = len(TEXT.vocab)
emb_dim = 50
hidden_dim = 50
out_dim = 2
lr = 1e-2
dropout = 0.2
model = BiLSTM(vocab_size, hidden_dim, emb_dim, out_dim, bsize, dropout, gpu=gpu)
if gpu:
    model.cuda()

import tqdm
optimizer = optim.Adam(model.parameters(), lr=lr)
lossf = nn.NLLLoss()
ep = 5
train_loss = []
valid_loss = []

for epoch in range(ep):
    avg_loss,acc = train(model, train_it, lossf, optimizer)
    tqdm.write('Train: loss %.2f acc %.1f' % (avg_loss, acc*100))

