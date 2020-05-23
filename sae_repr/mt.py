"""
模型训练模块
"""

import time

import torch


def train_model(model, data_loader, callback, *, model_name, model_path,
                num_epochs, learning_rate, loss, before_train=None, after_epoch=None,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                wd=0):
    is_load = input('是否加载上次%s模型(y/n)？' % model_name)
    if is_load == 'y':
        model.load_state_dict(torch.load(model_path))
        print(model, '\n')
    else:
        print(model, '\n')

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
        print(device)
        model = model.to(device)

        if before_train is not None:
            before_train(device)
        for epoch in range(num_epochs):
            l_sum, n, start = 0.0, 0, time.time()
            for data in data_loader:
                l = callback(model, device, data, loss)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                l_sum += l.cpu().item()
                n += 1
            print('epoch %d, loss %.4f, time %.1f sec' %
                  (epoch + 1, l_sum / n, time.time() - start))
            if after_epoch is not None:
                is_break = after_epoch(model, epoch, l_sum / n)
                if is_break:
                    break

        is_save = input('\n是否存储本次%s模型(y/n)？' % model_name)
        if is_save == 'y':
            torch.save(model.state_dict(), model_path)
