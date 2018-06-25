import torch
import torch.nn as nn


class MixedNet(nn.Module):
    dim_description = 8

    def __init__(self, description_voc_size, text_length=3200):
        super().__init__()

        self.description_emb = nn.Embedding(description_voc_size, self.dim_description)
        self.description_emb.weight.data.uniform_(-0.01, 0.01)

        self.text_length = text_length
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
                out_channels=8,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=4, padding=1),
            nn.Conv1d(
                in_channels=8,
                out_channels=1,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=4, padding=1),
        )
        self.relu = nn.ReLU()
        self.out = nn.Linear(244, 1)

    def forward(self, features):
        cat_features = features[:, :-self.text_length].float()
        text_features = features[:, -self.text_length:]
        x = self.description_emb(text_features)
        conv_out = self.layers(x.transpose(1, 2))

        batch_size = features.size(0)

        final_input = torch.cat([cat_features.view(batch_size, -1), conv_out.view(batch_size, -1)], dim=1)

        return self.out(self.relu(final_input)).unsqueeze(dim=2)
