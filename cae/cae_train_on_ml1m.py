import time
import math
import sys

sys.path.append('../sae_repr')

import torch
from torch import nn
import torch.utils.data as Data

from sae_repr.mt import train_model
from dataloader_ml1m import Ml1mRatingDataset
from dataloader_ml1m import get_ml1m_movie_map
from sae_train_on_ml1m import L1_none_zero_loss

num_movies = 3883
num_users = 6040
in_size = int(math.sqrt(num_movies)) + 1
batch_size = 128


class CAE(nn.Module):
    def __init__(self, channels, *, hidden_fn=nn.ReLU(), output_fn=None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=5),
            hidden_fn,
            nn.Conv2d(channels[0], channels[1], kernel_size=5),
            hidden_fn
        )
        decoder_list = [nn.ConvTranspose2d(channels[1], channels[0], kernel_size=5),
                        hidden_fn,
                        nn.ConvTranspose2d(channels[0], 1, kernel_size=5)]
        if output_fn is not None:
            decoder_list.append(output_fn)
        self.decoder = nn.Sequential(*decoder_list)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_cae_model(cae, data_loader, *, model_name='CAE', model_path='./cae.pt',
                    num_epochs=30, learning_rate=1e-3, loss_fn=L1_none_zero_loss,
                    test_rating=None, wd=1e-4):
    # dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev = torch.device('cpu')
    if test_rating is not None:
        test_rating = test_rating.to(dev)

    def sae_callback(model, device, data, loss):
        x = torch.cat((data, torch.zeros(data.shape[0], in_size ** 2 - num_movies)), dim=1) \
            .view(data.shape[0], 1, in_size, in_size).to(device)
        reconst = model(x)

        reconst[x == 0] = 0
        l = loss(reconst.view(data.shape[0], -1), x.view(data.shape[0], -1))

        return l

    def test(model, epoch, train_loss):
        start = time.time()
        reconst = model(test_rating)

        reconst[test_rating == 0] = 0
        l = loss_fn(reconst.view(num_users, -1), test_rating.view(num_users, -1)).item()

        print('epoch %d: test loss %.4f, time %.1f\n' % (epoch + 1, l, time.time() - start))

    train_model(cae, data_loader, sae_callback, model_name=model_name, model_path=model_path,
                num_epochs=num_epochs, learning_rate=learning_rate, loss=loss_fn, device=dev,
                after_epoch=test if test_rating is not None else None, wd=wd)


def main(*, channels, hidden_fn=nn.ReLU(), model_name='CAE', model_path='./cae.pt', wd=1e-4):
    movie_id2idx, movie_idx2id, movie_title_idx2t = get_ml1m_movie_map()
    train_loader = Data.DataLoader(dataset=Ml1mRatingDataset('../ml-1m/train_ratings.dat', movie_id2idx),
                                   batch_size=batch_size, shuffle=True)
    test_rating = Ml1mRatingDataset('../ml-1m/test_ratings.dat', movie_id2idx)
    test_rating.ratings = torch.cat((test_rating.ratings, torch.zeros(num_users, in_size ** 2 - num_movies)), dim=1) \
        .view(num_users, 1, in_size, in_size)

    cae = CAE(channels, hidden_fn=hidden_fn)
    train_cae_model(cae, train_loader, test_rating=test_rating.ratings, model_name=model_name,
                    model_path=model_path, wd=wd)


if __name__ == '__main__':
    channels = [16, 8]
    params = {'channels': channels, 'hidden_fn': nn.ReLU(),
              'model_name': 'CAE' + str(channels), 'model_path': './cae-%s.pt' % channels}
    main(**params)
