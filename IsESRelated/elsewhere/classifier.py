# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Callable, Iterator, Union, Optional, List
import numpy as np
import time
import os

class text_cnn(nn.Module):
    def __init__(self,
        vocab_size: int,
        embedding_dim: int=256,
        num_filters: int=128,
        filter_sizes=(3,4,5),
        padding_index: int=0,
        dropout: float=0.2,
        embedding_freeze=False,
        word2vec_embeddings=None):
        super(text_cnn, self).__init__()
        if word2vec_embeddings is None:
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=padding_index)
        else:
            self.vocab_size = word2vec_embeddings.shape[0]
            self.embedding_dim = word2vec_embeddings.shape[1]
            self.embedding = nn.Embedding.from_pretrained(word2vec_embeddings.float(), freeze=embedding_freeze)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, self.embedding_dim)) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 2)
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        pool_size = int(x.size(2))
        x = F.max_pool1d(x, pool_size).squeeze(2)
        return x
    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out