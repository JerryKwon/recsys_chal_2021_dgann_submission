"""
Name: model.py
Description: Neural Network models for DAN and ANN
* DAN(Deep Averaging Network) with GloVe word embedding model
    └── WordDropout: Dropout for inputs(word vectors by GloVe model) of DAN model
* ANN
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Dropout for inputs(word vectors by GloVe model) of DAN model
class WordDropout(nn.Module):
    def __init__(self, embedding_layer, DEVICE, dropout=0.3):
        super(WordDropout, self).__init__()
        self.dropout = dropout
        self.DEVICE = DEVICE
        self.embedding_layer = embedding_layer

    def sampling(self, batch):
        new_batches = []
        inputs, len_idx = batch
        max_size = inputs.shape[1]

        target_idx = np.arange(max_size)[np.random.binomial(1, 0.3, max_size) == 1]

        embeddings = self.embedding_layer((inputs[:, target_idx] if self.training else inputs))
        for emb in embeddings:
          target_emb = emb[emb.sum(axis=1) != 0]

          if target_emb.nelement() == 0:
            new_batches.append(torch.zeros(20, dtype=torch.float).to(self.DEVICE))
          else:
            new_batches.append(target_emb.mean(axis=0))

        result = torch.vstack(new_batches)
        result.requires_grad = True
        return result

    def forward(self, x):
        return self.sampling(x)


# DAN(Deep Averaging Neural Network) with GloVe embedding model
class CustomDAN(nn.Module):
    def __init__(self, embeddings, DEVICE, hidden_layers=2):
        super(CustomDAN, self).__init__()

        self.embedding_layer = self.make_emb_layer(embeddings)
        self.word_dropout = WordDropout(self.embedding_layer, DEVICE)

        h_layers = []
        for idx in range(hidden_layers):
            h_layers.append(self.make_hidden_layer(20, 20))

        self.hidden_layers = nn.Sequential(*h_layers)

        self.softmax = nn.Softmax()

        self.dense_out = nn.Linear(20, 2)

    def make_emb_layer(self, embeddings):
        num_embeddings, embedding_dim = embeddings.shape
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({"weight": torch.FloatTensor(embeddings)})
        emb_layer.weight.requires_grad = False

        return emb_layer

    def make_hidden_layer(self, in_channel, out_channel, dropout=0.2):
        layers = []

        layers.append(nn.Linear(in_channel, out_channel))
        layers.append(nn.BatchNorm1d(out_channel))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        wd = self.word_dropout(x)

        h = self.hidden_layers(wd)

        x = self.dense_out(h)
        x = self.softmax(x)

        return h, x

# ANN model
class CustomANN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(44, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.leaky_relu(self.bn1((self.fc1(x))))
        x = F.leaky_relu(self.bn2((self.fc2(x))))
        x = F.leaky_relu(self.bn3((self.fc3(x))))
        x = self.fc4(x)
        x = torch.sigmoid(x)

        return x
