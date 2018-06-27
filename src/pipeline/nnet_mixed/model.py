import torch
import torch.nn as nn


class MixedNet(nn.Module):
    dim_city = 10

    def __init__(self, city_voc_size=1733):
        super().__init__()

        self.city_emb = nn.Embedding(city_voc_size, self.dim_city)
        self.city_emb.weight.data.uniform_(-0.01, 0.01)

        # self.customers_emb.weight.requires_grad = False
        # self.customers_emb.weight.data.copy_(torch.from_numpy(pretrained_weight))

        self.layers = nn.Sequential(
            nn.Linear(101, 99),
            nn.ReLU(inplace=True),
            nn.Linear(99, 1),
        )

    def forward(self, features):
        city_le = features[:, -1:].long()
        city_emb_values = self.city_emb(city_le).squeeze(dim=1)

        net_input = torch.cat([features[:, :-1], city_emb_values], dim=1)

        return self.layers(net_input).unsqueeze(dim=2)
