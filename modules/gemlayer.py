import torch
import torch.nn as nn
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        temp1 = x.clamp(min=self.eps).pow(self.p)

        temp1 = temp1.unsqueeze(1)

        temp = nn.AdaptiveAvgPool1d(x[0].size())(temp1)

        return temp.pow(1. / self.p)