#coding:utf-8
import sys
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
LABEL = data.LabelField(dtype=torch.float)

dataset = data.TabularDataset(path='text/sent.csv', format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
train, val, test = dataset.split(split_ratio=[0.7, 0.1, 0.2], random_state=random.getstate())
#print (len(train), len(val), len(test))
TEXT.build_vocab(train)
LABEL.build_vocab(train)
#print (LABEL.vocab.freqs.most_common(10))

bsize = 4
gpu = True
device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')

train_it, valid_it, test_it = data.BucketIterator.splits((train, val, test), batch_sizes=(bsize,bsize,bsize), device=device, sort_key=lambda x: len(x.text), repeat=False)

'''
for b in train_it:
    #print (b.text, b.label)
    print (b.label.data) is same as 
    print (b.label)
    sys.exit()

#is same as 
for idb, batch in enumerate(train_it):
    print (batch.text, '####', batch.label)

batch = next(iter(train_it))
print (batch)
print (batch.text)
print (batch.label)
print (len(train_it))
'''

def binary_accuracy(pred, label):
    rounded_pred = torch.round(torch.sigmoid(pred))
    correct = (rounded_pred == label).float()
    acc = correct.sum() / len(correct) 
    #print (correct, acc)
    return acc

def train(model, trainit, lossf, optimizer):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0
    for batch in trainit:
        optimizer.zero_grad()
        sent, label = batch.text, batch.label
        pred = model(sent).squeeze(1)
        loss = lossf(pred, label)
        acc = binary_accuracy(pred, label)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss/ len(train_it), epoch_acc/ len(train_it)
    
def evaluate(model, it, lossf):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for batch in it:
            sent, label = batch.text, batch.label
            #print (sent, sent.squeeze(1))
            pred = model(sent)
            if len(batch) != 1:
                pred = pred.squeeze(1)
            loss = lossf(pred, label)
            acc = binary_accuracy(pred, label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss/ len(valid_it), epoch_acc/len(valid_it)
    

vocab_size = len(TEXT.vocab)
emb_dim = 50
hidden_dim = 50
out_dim = 1
lr = 1e-2
nlayers = 2
bidir = True
dropout = 0.5
model = BiLSTM(vocab_size, hidden_dim, emb_dim, out_dim, bsize, nlayers, bidir, dropout, gpu=gpu)

optimizer = optim.Adam(model.parameters()) #no need to specify LR for adam
lossf = nn.BCEWithLogitsLoss()
ep = 5

if gpu:
    model.to(device)
    lossf.to(device)

for epoch in range(ep):
    tr_loss, tr_acc = train(model, train_it, lossf, optimizer)
    vl_loss, vl_acc = evaluate(model, valid_it, lossf)
    print('TRAIN: loss %.2f acc %.1f' % (tr_loss, tr_acc*100)) 
    print('VALID: loss %.2f acc %.1f' % (vl_loss, vl_acc*100))
    te_loss, te_acc = evaluate(model, test_it, lossf)
    print('TEST: loss %.2f acc %.1f' % (te_loss, te_acc*100))


