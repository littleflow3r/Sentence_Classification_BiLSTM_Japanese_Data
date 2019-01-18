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
LABEL = data.Field(sequential=False)

dataset = data.TabularDataset(path='text/sent.csv', format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
train, val, test = dataset.split(split_ratio=[0.7, 0.1, 0.2], random_state=random.getstate())
#print (len(train), len(val), len(test))
TEXT.build_vocab(train, val, test)
LABEL.build_vocab(train, val, test)
#print (LABEL.vocab.freqs.most_common(10))

bsize = 4
gpu = False
device = 'cpu'
if gpu and torch.cuda.is_available:
    device = torch.device('cuda:1')

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
        label.sub_(1) #substract with 1, because the label tensor using index where index 1=value 0, 2=1
        truth_res += list(label.cpu().numpy())
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
    #print ('truth pred', truth_res, pred_res)
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

'''
for epoch in range(1, ep+1):
    t_loss = 0.0
    model.train()
    for x, y in tqdm.tqdm(train_batch_it):
        optimizer.zero_grad()
        pred = model(x)
        pred = pred.data.cpu().max(1)[1]
        print ('PRED:', pred)
        print ('LABEL:', y)
        loss = lossf(pred, y)
        import sys
        sys.exit()
        loss.backward()
        opt.step()
        t_loss += loss.item() * x.size(0)
    epoch_loss = t_loss/ len(train)
    v_loss = 0.0
    model.eval()
    for x,y in valid_batch_it:
        pred = model(x)
        pred_label = pred.data.cpu().max(1)[1].numpy()
        loss = lossf(pred_label,y)
        val_loss += loss.item() * x.size(0)
    val_loss /= len(valid)
    train_loss.append(epoch_loss)
    valid_loss.append(val_loss)
    print ('Epoch: {}, Training loss: {:.4f}, Validation loss: {:.4f}'.format(epoch, epoch_loss, val_loss))
'''
