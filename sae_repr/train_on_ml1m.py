"""
在 ml1m 数据集上训练 SAE 模型和预测模型构建推荐系统
"""

import time

import torch
from torch import nn
import torch.utils.data as Data

import sae_repr_dataloader_ml1m as ml1m
from mt import train_model
from sae import SAE
from prediction import MLP


def train_sae_model(sae, data_loader, data_extract, *, model_name='SAE', model_path='./sae.pt',
                    num_epochs=5, learning_rate=1e-3, loss_fn=nn.MSELoss()):
    def sae_callback(model, device, data, loss):
        x = data_extract(data).to(device)
        encode, reconst = model(x)

        if model.num_users != 0:
            x = x[:, 1:]
        # reconst[x == 0] = 0
        l = loss(x, reconst)

        return l

    train_model(sae, data_loader, sae_callback, model_name=model_name, model_path=model_path,
                num_epochs=num_epochs, learning_rate=learning_rate, loss=loss_fn)


def train_regression(regression, ae, data_loader,
                     model_name='MLP', model_path='./mlp.pt'):
    def regression_callback(model, device, data, loss):
        user_movie, rating = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            x = ae.encode(user_movie)
        y = model(x).view(-1)

        return loss(rating, y)

    train_model(regression, data_loader, regression_callback, model_name=model_name,
                model_path=model_path, num_epochs=10, learning_rate=1e-3,
                loss=nn.MSELoss(), before_train=lambda device: ae.to(device))


if __name__ == '__main__':
    movie_features, movie_title_i2t, movie_idx2id, movie_id2idx = ml1m.get_ml1m_movie_data()
    print('电影数量:', len(movie_features))
    print('最大电影id:', movie_idx2id[-1])
    print('电影特征长度:', movie_features.shape[1])
    print('第一个电影特征:', movie_features[0], '\n')

    # user_feature_type
    # 0 - 只有 id 的 one-hot 编码；
    # 1 - id 和其他特征的 one-hot 编码；
    # 2 - id 为数字；
    # 3 - 没有 id 类特征
    user_feature_type = 1
    user_features = ml1m.get_ml1m_user_features(user_feature_type)
    print('用户数:', len(user_features))
    print('用户特征长度:', user_features.shape[1], '\n')

    ratings = ml1m.get_ml1m_ratings()
    cut_point = int(len(ratings) * 0.8)
    train_ratings = ratings[:cut_point]
    test_ratings = ratings[cut_point:]
    print('评分数:', len(ratings))
    print('第一个评分:', ratings[0], '\n')

    batch_size = 128
    num_users = 6040
    hidden_size = 32

    train_loader = Data.DataLoader(dataset=
                                   ml1m.Ml1mItemDataset(train_ratings, user_features,
                                                        movie_features, movie_id2idx),
                                   batch_size=batch_size, shuffle=False)
    sae = SAE([user_features.shape[1] + movie_features.shape[1], hidden_size],
              hidden_fn=nn.Sigmoid())
    train_sae_model(sae, train_loader, lambda d: d[0])

    train_loader = Data.DataLoader(dataset=
                                   ml1m.Ml1mItemDataset(train_ratings, user_features,
                                                        movie_features, movie_id2idx),
                                   batch_size=batch_size, shuffle=False)
    regression = MLP([hidden_size, hidden_size // 2, 1])
    train_regression(regression, sae, train_loader)

    print('\n在测试集上运行')
    test_loader = Data.DataLoader(dataset=
                                  ml1m.Ml1mItemDataset(test_ratings, user_features,
                                                       movie_features, movie_id2idx),
                                  batch_size=batch_size, shuffle=False)
    l_sum, n, start = 0.0, 0, time.time()
    loss = nn.MSELoss()
    compare_list = []
    # 将模型都移到 cpu 上面
    cpu = torch.device('cpu')
    sae = sae.to(cpu)
    regression = regression.to(cpu)
    for user_movie, rating in test_loader:
        x = sae.encode(user_movie)
        y = regression(x).view(-1)
        l = loss(rating, y)
        l_sum += l.item()
        n += 1

        compare_list.clear()
        for i in range(30):
            compare_list.append((rating[i].item(), y[i].item()))
    print('loss %.4f, time %.1f sec' % (l_sum / n, time.time() - start))
    print('\n随机抽取 %d 个评分：' % len(compare_list))
    for rating, y in compare_list:
        print('%d - %.1f' % (rating, y))
