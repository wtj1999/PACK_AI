import torch
import torch.nn as nn

class DeepSetModel(nn.Module):
    def __init__(self, in_dim, emb_dim=64, node_dim=128, out_dim=1):
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.phi = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, out_dim),
        )

        self.rho = nn.Sequential(
            nn.Linear(node_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, node_dim),
            # nn.ReLU(),
        )

        self.linear1 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        # x: (B, n, in_dim)
        B, n, d = x.shape
        emb = self.phi(x)  # [b, n, 1]
        emb = emb.transpose(1, 2)  # [b, 1, n]
        emb = self.rho(emb)  # [b, 1, n]
        emb = emb.transpose(1, 2) # [b, n, 1]
        # emb = self.linear1(emb)


        return emb