import random
import torch
import torch.nn as nn


class JensenShannonInfoMax(nn.Module):
    """discriminator network.
    Args:
        x_dim (int): input dim, for example m x n x c for [m, n, c]
        y_dim (int): dimension of latent code (typically a number in [10 - 256])
    """

    def __init__(self, x_preprocessor: nn.Module, y_preprocessor: nn.Module, discriminator: nn.Module):
        super(JensenShannonInfoMax, self).__init__()
        self.discriminator = discriminator
        self.x_preprocessor = x_preprocessor
        self.y_preprocessor = y_preprocessor
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x_raw, y_raw):
        """
        Inputs:
            x : input from train_loader (batch_size x input_size )
            z : latent codes associated with x (batch_size x z_dim)
        """
        x, y = x_raw, y_raw # identity
        if self.x_preprocessor is not None:
            x = self.x_preprocessor(x_raw)
        if self.y_preprocessor is not None:
            y = self.y_preprocessor(y_raw)

        # Generate fake data by shifting
        shift = random.randint(1, x.size(0) - 1)
        fake_x = torch.cat([x[-shift:], x[:-shift]], dim=0)
        fake_z = torch.cat([y[-shift:], y[:-shift]], dim=0)

        true_logits = self.discriminator(x, y)
        true_labels = torch.ones_like(true_logits)

        fake_z_logits = self.discriminator(x, fake_z)
        fake_x_logits = self.discriminator(fake_x, y)
        fake_logits = torch.cat([fake_z_logits, fake_x_logits], dim=0)
        fake_labels = torch.zeros_like(fake_logits)

        true_loss = self.bce_loss(true_logits, true_labels)
        fake_loss = 0.5 * self.bce_loss(fake_logits, fake_labels)
        loss_info = true_loss + fake_loss
        return loss_info
