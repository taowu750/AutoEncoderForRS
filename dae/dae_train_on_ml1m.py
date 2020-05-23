import sys

sys.path.append('../sae')

from torch import nn

from sae.sae_train_on_ml1m import main

if __name__ == '__main__':
    NUM_USERS = 6040
    NUM_MOVIES = 3883

    hidden_layers = [256, 256]
    params = {'model_name': 'DAE' + str(hidden_layers), 'layers': [NUM_MOVIES, *hidden_layers],
              'hidden_fn': nn.Sigmoid(), 'model_path': './dae-%s.pt' % hidden_layers,
              'tied_weight': False}
    main(**params)
