import sys

sys.path.append('../sae_repr')

from torch import nn
import torch.utils.data as Data

from vae import VAE
from vae_train_on_ml1m import train_vae_model
from dataloader_jester import JesterRatingDataset


def main(*, layers, hidden_fn=nn.ReLU(), model_name='VAE', model_path='./vae.pt', wd=1e-4, num_epochs=30):
    batch_size = 128

    train_loader = Data.DataLoader(dataset=JesterRatingDataset('../jester/train_ratings.csv'),
                                   batch_size=batch_size, shuffle=True)
    print('已加载训练集数据...')
    test_rating = JesterRatingDataset('../jester/test_ratings.csv')
    print('\n已加载测试集数据...')

    vae = VAE(*layers, hidden_fn=hidden_fn)
    train_vae_model(vae, train_loader, test_rating=test_rating.ratings, model_name=model_name,
                    model_path=model_path, wd=wd, num_epochs=num_epochs)


if __name__ == '__main__':
    NUM_JOKES = 100

    hidden_layers = [32, 32]
    params = {'model_name': 'VAE' + str(hidden_layers), 'layers': [NUM_JOKES, *hidden_layers],
              'hidden_fn': nn.ReLU(), 'model_path': './vae-%s.pt' % hidden_layers,
              'num_epochs': 60}
    main(**params)
