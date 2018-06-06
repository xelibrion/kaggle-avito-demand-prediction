import torch
import torch.nn as nn


class MixedNet(nn.Module):
    dim_description = 8

    def __init__(self, description_voc_size):
        super().__init__()

        self.description_emb = nn.Embedding(description_voc_size, self.dim_description)
        self.description_emb.weight.data.uniform_(-0.01, 0.01)
        # self.customers_emb.weight.requires_grad = False
        # self.customers_emb.weight.data.copy_(torch.from_numpy(pretrained_weight))

    def forward(self, features):
        return
