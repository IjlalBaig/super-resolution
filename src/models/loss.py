from torch import nn


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, real_label, fake_label):
        loss_disc = (fake_label ** 2 + (real_label - 1) ** 2).mean()
        loss_gen = ((fake_label - 1) ** 2).mean()
        return loss_gen, loss_disc