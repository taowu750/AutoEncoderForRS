"""
进行分类或预测的后端模块
"""

import torch
from torch import nn


class FM(nn.Module):
    def __init__(self, num_feature, k):
        super().__init__()
        self.f = num_feature
        self.k = k
        self.linear = nn.Linear(self.f, 1)
        self.v = nn.Parameter(torch.rand(num_feature, k), requires_grad=True)

    def forward(self, x):
        linear_part = self.linear(x)
        inter_part1 = torch.mm(x, self.v)
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))
        inter_part = torch.sum(torch.sub(torch.pow(inter_part1, 2), inter_part2), dim=1)

        return linear_part + (0.5 * inter_part).view(-1, 1)


class MLP(nn.Module):
    def __init__(self, layers, output_fn=None):
        super().__init__()
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i + 1]))
            if i != len(layers) - 2:
                layer_list.append(nn.ReLU())
        if output_fn is not None:
            layer_list.append(output_fn())
        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)
