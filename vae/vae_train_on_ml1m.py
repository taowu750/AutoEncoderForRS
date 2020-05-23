import time
import sys

sys.path.append('../sae_repr')

import torch
from torch import nn
import torch.utils.data as Data

from vae import VAE
from sae_repr.mt import train_model
from sae_dataloader_ml1m import Ml1mRatingDataset
from sae_dataloader_ml1m import get_ml1m_movie_map
from sae_train_on_ml1m import L1_none_zero_loss


batch_size = 128


def train_vae_model(vae, data_loader, *, model_name='VAR', model_path='./vae.pt',
                    num_epochs=30, learning_rate=1e-3, loss_fn=L1_none_zero_loss,
                    test_rating=None, wd=1e-4):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if test_rating is not None:
        test_rating = test_rating.to(dev)

    def sae_callback(model, device, data, loss):
        x = data.to(device)
        reconst, mu, log_var = model(x)

        reconst[x == 0] = 0
        l = loss(reconst, x)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
        l += kl_div

        return l

    def test(model, epoch, train_loss):
        start = time.time()
        reconst, *_ = model(test_rating)

        reconst[test_rating == 0] = 0
        l = loss_fn(reconst, test_rating).item()

        print('epoch %d: test loss %.4f, time %.1f\n' % (epoch + 1, l, time.time() - start))

    train_model(vae, data_loader, sae_callback, model_name=model_name, model_path=model_path,
                num_epochs=num_epochs, learning_rate=learning_rate, loss=loss_fn, device=dev,
                after_epoch=test if test_rating is not None else None, wd=wd)


def main(*, layers, hidden_fn=nn.ReLU(), model_name='VAE', model_path='./vae.pt', wd=1e-4):
    movie_id2idx, movie_idx2id, movie_title_idx2t = get_ml1m_movie_map()
    train_loader = Data.DataLoader(dataset=Ml1mRatingDataset('../ml-1m/train_ratings.dat', movie_id2idx),
                                   batch_size=batch_size, shuffle=True)
    test_rating = Ml1mRatingDataset('../ml-1m/test_ratings.dat', movie_id2idx)

    sae = VAE(*layers, hidden_fn=hidden_fn)
    train_vae_model(sae, train_loader, test_rating=test_rating.ratings, model_name=model_name,
                    model_path=model_path, wd=wd)


if __name__ == '__main__':
    NUM_MOVIES = 3883

    hidden_layers = [512, 512]
    params = {'model_name': 'VAE' + str(hidden_layers), 'layers': [NUM_MOVIES, *hidden_layers],
              'hidden_fn': nn.Sigmoid(), 'model_path': './vae-%s.pt' % hidden_layers}
    main(**params)
