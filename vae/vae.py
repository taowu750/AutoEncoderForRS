import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, in_features, h_dim, z_dim, *, hidden_fn=nn.ReLU(),
                 output_fn=None):
        super().__init__()
        self.hidden_fn = hidden_fn
        self.encoder_hidden = nn.Linear(in_features, h_dim)
        self.mu = nn.Linear(h_dim, z_dim)
        self.log_var = nn.Linear(h_dim, z_dim)
        decoder_list = [nn.Linear(z_dim, h_dim),
                        hidden_fn,
                        nn.Linear(h_dim, in_features)]
        if output_fn is not None:
            decoder_list.append(output_fn)
        self.decoder = nn.Sequential(*decoder_list)

    # 编码，学习均值和方差
    def encode_mu_var(self, x):
        h = self.hidden_fn(self.encoder_hidden(x))
        return self.mu(h), self.log_var(h)

    # 将高斯分布均值与方差重表示，生成隐变量 z
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.rand_like(std)
        return mu + eps * std

    # 编码，生成隐藏变量
    def encode(self, x):
        return self.reparameterize(*self.encode_mu_var(x))

    # 解码，生成重构
    def decode(self, z):
        return self.decoder(z)

    # 计算重构值和隐变量z的分布参数
    def forward(self, x):
        mu, log_var = self.encode_mu_var(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)

        return x_reconst, mu, log_var
