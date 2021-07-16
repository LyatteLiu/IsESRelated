# -*- coding: utf-8 -*-
from .elsewhere.tokenizer import punk_tokenizer
from .elsewhere.embedding import get_numpy_array_from_torch_tensor
from .elsewhere.classifier import text_cnn

import torch
import numpy as np
import re
from typing import Callable, Iterator, Union, Optional, List

tokenizer = None
is_es_related_model = None
t_device = None
max_length = 3500

def load_is_es_related_model(model_path: str='./IsESRelated/ml_data/text_cnn_fold_1.model', vocab_path: str='./IsESRelated/ml_data/vocab_file.txt'):
    global is_es_related_model
    global tokenizer
    global t_device
    if torch.cuda.is_available():
    #if False:
        t_device = torch.device('cuda')
    else:
        t_device = torch.device('cpu')
    if tokenizer is None:
        tokenizer = punk_tokenizer.from_vocab_file(vocab_path)
    model = text_cnn(vocab_size=len(tokenizer.vocab_list))
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    assert len(tokenizer.vocab_list) == model.vocab_size
    model.eval()
    model.to(t_device) # 
    is_es_related_model = model

def get_is_es_related_prediction(subject: str, body: str):
    text = subject + ' ' + body # big string big object allocation, you imbecile!
    ids = tokenizer.tokenize(text)
    while len(ids) < 12: # TODO: avoid these two hardcode
        ids.append(0)
    ids = ids[:3500]
    tx = torch.LongTensor(ids).unsqueeze(0)
    tx = tx.to(t_device)
    y_pred = is_es_related_model(tx) # y_pred 'd be(1,2)
    ny = get_numpy_array_from_torch_tensor(y_pred)
    ny = ny - np.max(ny,axis=1)
    ny = np.exp(ny)
    ny /= np.sum(ny,axis=1)
    return ny[0,1]

# init
load_is_es_related_model()