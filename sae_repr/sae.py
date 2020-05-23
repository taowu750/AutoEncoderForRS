"""
SAE 实现
"""

import torch
import torch.nn as nn


# 在自编码器中使用的绑定权重
class TiedWeight(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.params = nn.Parameter(torch.rand(in_features, out_features), requires_grad=True)
        if bias:
            self.bias_param = nn.Parameter(torch.rand(out_features), requires_grad=True)

    def forward(self, x, is_forward=True):
        if is_forward:
            result = torch.mm(x, self.params)
            if self.bias:
                result += self.bias_param
        else:
            result = x.clone()
            if self.bias:
                result += self.bias_param
            result = torch.mm(result, self.params.t())

        return result

    def __str__(self):
        return 'TiedWeight(in_features=%d, out_features=%d, bias=%s)' % \
               (self.in_features, self.out_features, self.bias)


class SAE(nn.Module):
    # 如果 num_users 不为 0，表示要用到用户的独立特征向量。此时输入数据第一列是用户 id
    def __init__(self, layers, *, tied_weight=False, num_users=0,
                 hidden_fn=nn.ReLU(), output_fn=None):
        super().__init__()

        # 使用绑定权重的自编码器
        self.num_users = num_users
        self.tied_weight = tied_weight

        if num_users != 0:
            # 减 1 减去 id 特征
            layers[0] -= 1
            # 为每个用户添加独立的特征向量
            self.user_vectors = nn.Parameter(torch.rand(num_users, layers[1]), requires_grad=True)

        if tied_weight:
            self.encoder = nn.ModuleList()
            for i in range(len(layers) - 1):
                self.encoder.append(TiedWeight(layers[i], layers[i + 1]))
                self.encoder.append(hidden_fn)

            self.decoder = nn.ModuleList()
            i = 0
            for layer in self.encoder[::-1]:
                if isinstance(layer, TiedWeight):
                    self.decoder.append(layer)
                    if i != len(layers) - 2:
                        self.decoder.append(hidden_fn)
                    i += 1
            if output_fn is not None:
                self.decoder.append(output_fn)
        else:
            self.encoder = nn.ModuleList()
            for i in range(len(layers) - 1):
                self.encoder.append(nn.Linear(layers[i], layers[i + 1]))
                self.encoder.append(hidden_fn)

            layers.reverse()
            decoder_layers = []
            for i in range(len(layers) - 1):
                decoder_layers.append(nn.Linear(layers[i], layers[i + 1]))
                if i != len(layers) - 2:
                    decoder_layers.append(hidden_fn)
            if output_fn is not None:
                decoder_layers.append(output_fn)
            self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        if self.num_users != 0:
            # 输入第一列是用户 id
            if len(x.shape) > 1:
                id = x[:, 0].long()
                features = x[:, 1:]
            else:
                id = x[0].long()
                features = x[1:]

            first_hidden = self.encoder[0](features) + self.user_vectors[id - 1]
            result = first_hidden
            for layer in self.encoder[1:]:
                result = layer(result)
        else:
            result = x
            for layer in self.encoder:
                result = layer(result)

        return result

    def decode(self, x):
        if self.tied_weight:
            result = x
            for layer in self.decoder:
                if isinstance(layer, TiedWeight):
                    result = layer(result, is_forward=False)
                else:
                    result = layer(result)
        else:
            result = self.decoder(x)

        return result

    def forward(self, x):
        enc = self.encode(x)
        dec = self.decode(enc)

        return enc, dec
