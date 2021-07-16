# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch

class word2vec_neg_sampling(nn.Module):

    def __init__(self, embedding_size, vocab_size, device, noise_dist=None, negative_samples=10):
        super(word2vec_neg_sampling, self).__init__()

        self.embeddings_input = nn.Embedding(vocab_size, embedding_size)
        self.embeddings_context = nn.Embedding(vocab_size, embedding_size)
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples
        self.device = device
        self.noise_dist = noise_dist

        # Initialize both embedding tables with uniform distribution
        self.embeddings_input.weight.data.uniform_(-1, 1)
        self.embeddings_context.weight.data.uniform_(-1, 1)

    def forward(self, input_word, context_word):
        debug = not True
        if debug:
            print('input_word.shape: ', input_word.shape)  # bs
            print('context_word.shape: ', context_word.shape)  # bs
        batch_size = context_word.shape[0]
        # computing out loss
        emb_input = self.embeddings_input(input_word)  # bs, emb_dim

        emb_context = self.embeddings_context(context_word)  # bs, emb_dim
        emb_product = torch.mul(emb_input, emb_context)  # bs, emb_dim # 点乘是一种粗略的相似度计算，相似性越大，点积越大
        emb_product = torch.sum(emb_product, dim=1)  # bs
        out_loss = F.logsigmoid(emb_product)  # bs

        if self.negative_samples > 0:
            # computing negative loss
            if self.noise_dist is None:
                noise_dist = torch.ones(self.vocab_size)
            else:
                noise_dist = self.noise_dist
            num_neg_samples_for_this_batch = batch_size * self.negative_samples  # 负采样的个数
            negative_example = torch.multinomial(noise_dist, num_neg_samples_for_this_batch,  # 抽样取值
                                                 replacement=True)
            negative_example = negative_example.view(batch_size, self.negative_samples).to(
                self.device)  # bs, num_neg_samples
            emb_negative = self.embeddings_context(negative_example)  # bs, neg_samples, emb_dim
            emb_product_neg_samples = torch.bmm(emb_negative.neg(), emb_input.unsqueeze(2))  # bs, neg_samples, 1
            noise_loss = F.logsigmoid(emb_product_neg_samples).squeeze(2).sum(1)  # bs
            total_loss = -(out_loss + noise_loss).mean()
            return total_loss

        else:
            return -(out_loss).mean()

def get_numpy_array_from_torch_tensor(x):
    if x.device.type.lower() != 'cpu':
        x = x.cpu()
    return x.detach().numpy()