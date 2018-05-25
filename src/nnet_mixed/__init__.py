import torch
import torch.nn as nn


class EmbeddingNet(nn.Module):
    dim_region = 8

    def __init__(self, region_voc_size):
        super().__init__()

        self.region_emb = nn.Embedding(region_voc_size, self.dim_region)
        self.region_emb.weight.data.uniform_(-0.01, 0.01)
        # self.customers_emb.weight.requires_grad = False
        # self.customers_emb.weight.data.copy_(torch.from_numpy(pretrained_weight))

    def forward(self, features):
        customers, content_variants = cats[:, 0], cats[:, 1]
        embeddings_tensor = torch.cat(
            [
                self.customers_emb(customers),
                self.content_emb(content_variants),
            ], dim=1)
        x = self.drop1(embeddings_tensor)
        x = self.drop2(F.relu(self.lin1(x)))
        return F.sigmoid(self.lin2(x))


def main():

    region_emb = nn.Embedding(10, 3)
    region_emb.weight.data.uniform_(-0.01, 0.01)

    features = torch.LongTensor(np.random.randint(2, size=(10, 7)))
    region = features[:, 1]

    df = pd.read_csv('input/train.csv')

    CATEGORICAL = {
        'region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3',
        'user_type'
    }
