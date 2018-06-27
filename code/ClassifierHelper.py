import random, os
from tqdm import *

import torch
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

class Classifier(nn.Module):
    def __init__(self, use_cuda=False):
        super(Classifier, self).__init__()
        self.use_cuda = use_cuda
        self.total_loss = []
        self.val_loss = []
        self.start_epoch = 0
        self.loss_function = nn.NLLLoss()

        if self.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, instance, parameters):
        pass

    def load_model(self, checkpt_file):
        print("=> loading checkpoint '{}'".format(checkpt_file))
        checkpoint = torch.load(checkpt_file, map_location=lambda storage, loc: storage)

        self.start_epoch = checkpoint['epoch']
        self.total_loss = checkpoint['total_loss']
        self.val_loss = checkpoint['val_loss']
        self.load_state_dict(checkpoint['state_dict'])
        self.load_params(checkpoint)

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpt_file, checkpoint['epoch']))

    def load_params(self, checkpoint):
        pass

    def save_model(self, checkpt_prefix, params):
        print("=> saving checkpoint '{}'".format(checkpt_prefix))
        torch.save(params, self.checkpt_file(checkpt_prefix))

    def checkpt_file(self, checkpt_prefix):
        return '{}.mdl'.format(checkpt_prefix)

    def run_training(self, n_epochs, refer_dataset, checkpt_prefix, parameters=None, learning_rate=0.001, batch_size=4):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate)

        refer_dataset.active_split = 'train'

        if self.use_cuda:
            dataloader = DataLoader(refer_dataset, batch_size, shuffle=True)
        else:
            dataloader = DataLoader(refer_dataset, batch_size, shuffle=True, num_workers=4)

        for epoch in range(self.start_epoch, n_epochs):
            self.train()
            refer_dataset.active_split = 'train'
            self.total_loss.append(0)

            for i_batch, sample_batched in enumerate(tqdm(dataloader, desc='{}rd epoch'.format(epoch))):
                instances, targets = self.trim_batch(sample_batched)
                self.clear_gradients(batch_size)

                label_scores = self(instances, parameters)
                loss = self.loss_function(label_scores.permute(1, 2, 0), targets)

                loss.backward()
                optimizer.step()

                self.total_loss[epoch] += loss.item()

            self.total_loss[epoch] = self.total_loss[epoch] / float(i_batch)

            self.save_model(checkpt_prefix, {
                'epoch': epoch + 1,
                'state_dict': self.state_dict(),
                'total_loss': self.total_loss,
                'val_loss': self.val_loss})

            print('Average training loss:{}'.format(self.total_loss[epoch]))

            if epoch % 10 == 0:
                self.val_loss.append(0)
                self.val_loss[-1] = self.run_testing(refer_dataset, 'val', parameters)
                print('Average validation loss:{}'.format(self.total_loss[epoch]))

        return self.total_loss

    def run_testing(self, refer_dataset, split=None, parameters=None):
        self.eval()
        refer_dataset.active_split = split
        n = len(refer_dataset)
        dataloader = DataLoader(refer_dataset)

        total_loss = 0
        for k, instance in enumerate(tqdm(dataloader, desc='Validation')):
            with torch.no_grad():
                label_scores = self(instance, parameters)
                targets = self.targets(instance)
                total_loss += self.loss_function(label_scores, targets)
        return total_loss/float(n)

    def trim_batch(self, instance):
        pass

    def clear_gradients(self, batch_size=None):
        self.zero_grad()

    def make_prediction(self, instances, parameters):
        self.eval()
        predictions = []
        with torch.no_grad():
            for i in tqdm(range(len(instances))):
                instance = instances[i]
                tag_scores = self(instance, parameters)
                val, index = tag_scores.data.max(0)
                predictions.append((index[0], val[0]))
        return predictions
