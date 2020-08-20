import time
import math
import sys

sys.path.append('../sae_repr')

import torch
from torch import nn
import torch.utils.data as Data

from cae_train_on_ml1m import CAE, train_cae_model
from dataloader_jester import JesterRatingDataset


NUM_USERS = 48483
NUM_JOKES = 100
in_size = int(math.sqrt(NUM_USERS)) + 1


def main(*, channels, hidden_fn=nn.ReLU(), model_name='CAE', model_path='./cae.pt', wd=1e-4):
    batch_size = 128

    train_loader = Data.DataLoader(dataset=JesterRatingDataset('../jester/train_ratings.csv'),
                                   batch_size=batch_size, shuffle=True)
    print('已加载训练集数据...')
    test_rating = JesterRatingDataset('../jester/test_ratings.csv')
    test_rating.ratings = torch.cat((test_rating.ratings, torch.zeros(NUM_USERS, in_size ** 2 - NUM_JOKES)), dim=1) \
        .view(NUM_USERS, 1, in_size, in_size)
    print('\n已加载测试集数据...')

    cae = CAE(channels, hidden_fn=hidden_fn)
    train_cae_model(cae, train_loader, test_rating=test_rating.ratings, model_name=model_name,
                    model_path=model_path, wd=wd)


if __name__ == '__main__':
    channels = [16, 8]
    params = {'channels': channels, 'hidden_fn': nn.ReLU(),
              'model_name': 'CAE' + str(channels), 'model_path': './cae-%s.pt' % channels}
    main(**params)
