import torch
import torch.nn as nn


class DiscriminatorNet(nn.Module):
    def __init__(self, d_model, z_dim):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(d_model + z_dim, d_model * 2),
            nn.BatchNorm1d(d_model * 2, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(d_model * 2, d_model * 2),
            nn.BatchNorm1d(d_model * 2, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(d_model * 2, d_model * 2),
            nn.BatchNorm1d(d_model * 2, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(d_model * 2, d_model * 2),
            nn.BatchNorm1d(d_model * 2, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(d_model * 2, d_model * 2),
            nn.BatchNorm1d(d_model * 2, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(d_model * 2, 1)
        )
        self.discriminator.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02) # N(0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, c_embeds, sampled_z):
        return self.discriminator(torch.cat([c_embeds, sampled_z], dim=-1))
