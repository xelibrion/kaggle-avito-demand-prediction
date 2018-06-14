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

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=self.dim_description,
                out_channels=8,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=8,
                out_channels=1,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=4, padding=1),
            nn.Linear(800, 1),
        )

    def forward(self, features):
        x = self.description_emb(features)
        return self.layers(x.transpose(1, 2))
