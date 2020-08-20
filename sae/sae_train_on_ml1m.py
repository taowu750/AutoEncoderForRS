import time
import sys

sys.path.append('../sae_repr')

import torch
from torch import nn
import torch.utils.data as Data

from sae_repr.sae import SAE
from sae_repr.mt import train_model
from dataloader_ml1m import Ml1mRatingDataset
from dataloader_ml1m import get_ml1m_movie_map


def L1_none_zero_loss(input, target):
    loss = (input - target).abs()
    return (loss.sum(dim=1) / (target != 0).sum(dim=1)).sum() / len(target)


def train_sae_model(sae, data_loader, *, model_name='SAE', model_path='./sae.pt',
                    num_epochs=30, learning_rate=1e-3, loss_fn=L1_none_zero_loss,
                    test_rating=None, wd=0):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if test_rating is not None:
        test_rating = test_rating.to(dev)

    def sae_callback(model, device, data, loss):
        x = data.to(device)
        encode, reconst = model(x)

        if model.num_users != 0:
            x = x[:, 1:]
        reconst[x == 0] = 0
        l = loss(reconst, x)

        return l

    def test(model, epoch, train_loss):
        start = time.time()
        encode, reconst = model(test_rating)
        if model.num_users != 0:
            reconst[test_rating[:, 1:] == 0] = 0
            l = loss_fn(reconst, test_rating[:, 1:]).item()
        else:
            reconst[test_rating == 0] = 0
            l = loss_fn(reconst, test_rating).item()

        print('epoch %d: test loss %.4f, time %.1f\n' % (epoch + 1, l, time.time() - start))

    train_model(sae, data_loader, sae_callback, model_name=model_name, model_path=model_path,
                num_epochs=num_epochs, learning_rate=learning_rate, loss=loss_fn, device=dev,
                after_epoch=test if test_rating is not None else None, wd=wd)


def main(*, layers, hidden_fn=nn.ReLU(), model_name='SAE', model_path='./sae.pt', wd=1e-4, num_users=0,
         add_uid=False, tied_weight=False):
    batch_size = 128

    movie_id2idx, movie_idx2id, movie_title_idx2t = get_ml1m_movie_map()
    train_loader = Data.DataLoader(dataset=Ml1mRatingDataset('../ml-1m/train_ratings.dat', movie_id2idx,
                                                             add_uid=add_uid),
                                   batch_size=batch_size, shuffle=True)
    test_rating = Ml1mRatingDataset('../ml-1m/test_ratings.dat', movie_id2idx, add_uid=add_uid)

    sae = SAE(layers, hidden_fn=hidden_fn, num_users=num_users, tied_weight=tied_weight)
    train_sae_model(sae, train_loader, test_rating=test_rating.ratings, model_name=model_name,
                    model_path=model_path, wd=wd)

    cpu = torch.device('cpu')
    sae = sae.to(cpu)
    test_rating.ratings = test_rating.ratings.to(cpu)
    for i in range(5):
        encode, reconst = sae(test_rating.ratings[i])
        print('\n用户 %d' % (i + 1))
        if add_uid:
            scores = test_rating.ratings[i][1:]
        else:
            scores = test_rating.ratings[i]
        for j, score in enumerate(scores):
            score = score.int().item()
            if score != 0:
                print('真实评分 %d, 预测评分：%.1f -《%s》' % (score, reconst[j].item(),
                                                    movie_title_idx2t[j]))
        print('\n')


if __name__ == '__main__':
    NUM_USERS = 6040
    NUM_MOVIES = 3883

    hidden_size = 512
    params = {'model_name': 'SAE%d' % hidden_size, 'layers': [NUM_MOVIES, hidden_size],
              'hidden_fn': nn.Sigmoid(), 'model_path': './sae-%d.pt' % hidden_size,
              'tied_weight': False}
    main(**params)
