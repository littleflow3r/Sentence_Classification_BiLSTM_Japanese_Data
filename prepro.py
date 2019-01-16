import os
import numpy as np
import csv

filedir = 'text/'
labels = []
texts = []
for label_class in ['dokujo-tsushin', 'it-life-hack']:
    dirname = os.path.join(filedir, label_class)
    for fname in os.listdir(dirname):
        if fname[-4:] == '.txt':
            with open(os.path.join(dirname, fname)) as f:
                texts.append(f.read())
            if label_class == 'dokujo-tsushin':
                labels.append(0)
            else:
                labels.append(1)


with open('text/sent.csv', 'w') as fw:
    writer = csv.writer(fw, quoting=csv.QUOTE_ALL)
    writer.writerow(['text','label'])
    writer.writerows(zip(texts, labels))

