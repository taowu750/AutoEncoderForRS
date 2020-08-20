import sys

sys.path.append('../sae')

from torch import nn

from sae.sae_train_on_jester import main


if __name__ == '__main__':
    NUM_USERS = 48483
    NUM_JOKES = 100

    hidden_layers = [32, 16]
    params = {'model_name': 'DAE' + str(hidden_layers), 'layers': [NUM_JOKES, *hidden_layers],
              'hidden_fn': nn.ReLU(), 'model_path': './dae-%s.pt' % hidden_layers,
              'num_epochs': 100}
    main(**params)
