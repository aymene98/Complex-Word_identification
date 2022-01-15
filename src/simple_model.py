import numpy as np
from nltk import RegexpTokenizer
import regex
from itertools import islice
import collections
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from util import read_data, word_context, read_decomposed_data, read_decomposed_data_and_equilibrate
import os
from evaluate_system import evaluate


'''
PREPROCESSING
'''


def SimpleWikiFrequencies(file):
    #wget https://github.com/LGDoor/Dump-of-Simple-English-Wiki/raw/master/corpus.tgz
    with open(file, encoding="utf-8") as fp:
        text = fp.read()
        toknizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
        words = toknizer.tokenize(text)
        filtered_words = [word.lower() for word in words if regex.search(r'(\p{Ll}|\d)', word.lower())]
        freq = collections.defaultdict(int)
        for word in filtered_words:
            freq[word] += 1
    return freq

frequencies = SimpleWikiFrequencies('../data/freq_corpora/simple_wikipedia.txt')

train_data_file = '../data/cwi_training/cwi_training.txt'
train_data_file_decomposed = '../data/cwi_training/cwi_training_allannotations.txt'
test_data_file = '../data/cwi_testing_annotated/cwi_testing_annotated.txt'
output_file = '../output/test.txt'

train_sentences, train_words, train_indexes, train_labels = read_data(train_data_file)
test_sentences, test_words, test_indexes, test_labels = read_data(test_data_file)

X = []
Y = []

for i in range(len(train_words)):
    w = train_words[i]
    y = train_labels[i]
    x = []
    if (w in frequencies.keys()) == True: #fréquence
        x.append(frequencies[w])
    else:
        x.append(-1)
    x.append(len(w))                      #longueur
    X.append(x)
    Y.append(int(y))

X = torch.tensor(X)
Y = torch.tensor(Y)
X = X.float()
Y = Y.long()


'''
APPRENTISSAGE
'''


train_set = TensorDataset(X, Y)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_1 = nn.Sequential(
            nn.Linear(2, 30),
            nn.ReLU(),
            nn.Linear(30,40),
            nn.ReLU(),
            nn.Linear(40,30),
            nn.ReLU(),
            nn.Linear(30,2)
        )
    def forward(self, x):
        return self.model_1(x)

def fit(model, epochs, train_loader):
    criterion = nn.CrossEntropyLoss(weight = torch.tensor([0.20, 0.80])) #donne plus d'importance aux mots complexes, peu nombreux comparés aux mots simples
    optimizer = optim.Adam(model.parameters())
    for epoch in range(epochs):
        total_loss = 0
        num = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            y_scores = model(x)
            loss = criterion(y_scores, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num += len(y)
        if epoch % (epochs // 10) == 0:
            print(epoch, total_loss / num)

mlp = MultiLayerPerceptron()
fit(mlp, 40, train_loader)


'''
TEST
'''


X_test = []
for i in range(len(test_words)):
    x = test_words[i]
    if (x in frequencies.keys()) == True:
        X_test.append([frequencies[x], len(x)]) #fréquence et longueur
    else:
        X_test.append([-1, len(x)])

X_test = torch.tensor(X)
X_test = X.float()

Y_score = mlp(X_test.float())
Y_pred = torch.max(Y_score, 1)[1]

np.savetxt(output_file, Y_pred.numpy(), delimiter='\n', fmt='%1.0d')
evaluate(test_data_file, output_file)