import time
import sys

sys.path.append('../sae_repr')

import torch
from torch import nn
import torch.utils.data as Data

from sae_repr.sae import SAE
from sae_repr.mt import train_model
from sae_train_on_ml1m import L1_none_zero_loss
from dataloader_jester import JesterRatingDataset


def train_sae_model(sae, data_loader, *, model_name='SAE', model_path='./sae.pt',
                    num_epochs=30, learning_rate=1e-3, loss_fn=L1_none_zero_loss,
                    test_rating=None, wd=0):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if test_rating is not None:
        test_rating = test_rating.to(dev)

    def sae_callback(model, device, data, loss):
        x = data.to(device)
        encode, reconst = model(x)
        reconst[x == 0] = 0
        l = loss(reconst, x)

        return l

    def test(model, epoch, train_loss):
        start = time.time()
        encode, reconst = model(test_rating)
        reconst[test_rating == 0] = 0
        l = loss_fn(reconst, test_rating).item()

        print('epoch %d: test loss %.4f, time %.1f\n' % (epoch + 1, l, time.time() - start))

    train_model(sae, data_loader, sae_callback, model_name=model_name, model_path=model_path,
                num_epochs=num_epochs, learning_rate=learning_rate, loss=loss_fn, device=dev,
                after_epoch=test if test_rating is not None else None, wd=wd)


def main(*, layers, hidden_fn=nn.ReLU(), model_name='SAE', model_path='./sae.pt',
         num_epochs=30, wd=1e-4, tied_weight=False):
    batch_size = 128

    train_loader = Data.DataLoader(dataset=JesterRatingDataset('../jester/train_ratings.csv'),
                                   batch_size=batch_size, shuffle=True)
    print('已加载训练集数据...')
    test_rating = JesterRatingDataset('../jester/test_ratings.csv')
    print('\n已加载测试集数据...')

    sae = SAE(layers, hidden_fn=hidden_fn, tied_weight=tied_weight)
    print('\n准备训练...\n')
    train_sae_model(sae, train_loader, test_rating=test_rating.ratings, model_name=model_name,
                    model_path=model_path, num_epochs=num_epochs, wd=wd)

    cpu = torch.device('cpu')
    sae = sae.to(cpu)
    test_rating.ratings = test_rating.ratings.to(cpu)
    for i in range(3):
        encode, reconst = sae(test_rating.ratings[i])
        print('\n用户 %d' % (i + 1))
        scores = test_rating.ratings[i]
        for j, score in enumerate(scores):
            score = score.item()
            if score != 0:
                print('真实评分 %.2f, 预测评分：%.2f' % (score, reconst[j].item()))
        print('\n')


if __name__ == '__main__':
    NUM_USERS = 48483
    NUM_JOKES = 100

    hidden_size = 32
    params = {'model_name': 'SAE%d' % hidden_size, 'layers': [NUM_JOKES, hidden_size],
              'hidden_fn': nn.ReLU(), 'model_path': './sae-%d.pt' % hidden_size,
              'num_epochs': 100}
    main(**params)
