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
        self.epoch = 0
        self.log_path = os.path.join(args.checkpoint_dir)
        self.loss_module = nn.ModuleList()
        self.log = {'val_accuracy': []}
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            loss_module = import_module('loss.' + loss_type)
            loss_function = getattr(loss_module, loss_type)(args)
            
            self.log['train_' + loss_type] = []
            self.log['val_' + loss_type] = []

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
            })

    def forward(self, projections, targets):
        loss_sum = 0
        for i, l in enumerate(self.loss):
            l['loss'] = l['function'](projections, targets)
            loss_sum += l['weight'] * l['loss']
            self.loss[i] = l
        return loss_sum
    
    def save_log(self, train_loss, val_loss, val_accuracy):
        self.epoch += 1
        for i, l in enumerate(self.loss):
            self.log['train_' + l['type']].append(train_loss)
            self.log['val_' + l['type']].append(val_loss)
        self.log['val_accuracy'].append(val_accuracy)

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def plot_loss(self, name='Log'):
        axis = np.linspace(1, self.epoch, self.epoch)
        fig, (loss_ax, acc_ax) = plt.subplots(nrows=2, sharex=True)

        plt.title(name)
        plt.grid(True)
        plt.xlabel('Epochs')
        loss_ax.set_ylabel('Loss')
        acc_ax.set_ylabel('Accuracy')
        for label, loss_value in self.log.items():
            if label == 'val_accuracy':
                acc_ax.plot(axis, loss_value, label=label)
            else:
                loss_ax.plot(axis, loss_value, label=label)

        # Add a legend to the plot
        lines1, labels1 = loss_ax.get_legend_handles_labels()
        lines2, labels2 = acc_ax.get_legend_handles_labels()
        fig.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.savefig(os.path.join(self.log_path, name + '.pdf'))
        plt.close(fig)
    
    def print_loss_coomponent(self):
        print('')
