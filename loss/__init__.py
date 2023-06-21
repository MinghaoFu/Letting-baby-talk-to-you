import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

from importlib import import_module

class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            loss_module = import_module(loss_type)
            loss_function = getattr(loss_module, loss_type)(args)

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function,
                'loss': 0
            })

    def forward(self, projections, targets):
        loss_sum = 0
        for i, l in enumerate(self.loss):
            l['loss'] = l['function'](projections, targets)
            loss_sum += l['weight'] * l['loss']
            self.loss[i] = l
        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)
