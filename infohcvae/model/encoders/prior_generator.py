import torch
import torch.nn as nn
import numpy as np


class PriorGenerator(nn.Module):
    def __init__(self, nzqdim, nzadim, nza_values):
        super(PriorGenerator, self).__init__()

        self.nzqdim = nzqdim
        self.nzadim = nzadim
        self.nza_values = nza_values

    def forward(self, c_ids):
        N = c_ids.size(0)
        # sample `zq`
        zq = torch.randn(N, self.nzqdim).to(c_ids.device)

        # sample `za`
        M = N * self.nzadim
        np_y = np.zeros((M, self.nza_values), dtype=np.float32)
        np_y[range(M), np.random.choice(self.nza_values, M)] = 1
        np_y = np.reshape(np_y, [M // self.nzadim, self.nzadim, self.nza_values])
        za = torch.from_numpy(np_y).to(c_ids.device)
        return zq, za
