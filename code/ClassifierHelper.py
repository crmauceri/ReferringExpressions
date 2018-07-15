import random, os
from tqdm import *

import torch
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

DEBUG = True

class Classifier(nn.Module):
    def __init__(self, disable_cuda=False):
        super(Classifier, self).__init__()
        self.total_loss = []
        self.val_loss = []
        self.start_epoch = 0
        self.loss_function = SequenceLoss(nn.CrossEntropyLoss)

        if not disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.use_cuda = True
        else:
            self.device = torch.device('cpu')
            self.use_cuda = False

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
        print("=> saving checkpoint '{}'".format(self.checkpt_file(checkpt_prefix)))
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
                self.clear_gradients(batch_size=targets.size()[0])

                loss = 0
                label_scores = self(instances, parameters)
                loss += self.loss_function(label_scores, targets)

                if DEBUG:
                    print([self.wordnet.ind2word[instances['vocab_tensor'][0, i]] for i in range(instances['vocab_tensor'].size()[1])])
                    print([self.wordnet.ind2word[torch.argmax(label_scores[0, i, :])] for i in range(instances['vocab_tensor'].size()[1]-1)])
                    print(loss)
                else:
                    loss.backward()
                    optimizer.step()

                self.total_loss[epoch] += loss.item()

            self.total_loss[epoch] = self.total_loss[epoch] / float(i_batch)

            self.clear_gradients(batch_size=1)
            self.save_model(checkpt_prefix, {
                'epoch': epoch + 1,
                'state_dict': self.state_dict(),
                'total_loss': self.total_loss,
                'val_loss': self.val_loss})

            print('Average training loss:{}'.format(self.total_loss[epoch]))

            if epoch % 10 == 0:
                self.val_loss.append(0)
                self.val_loss[-1] = self.run_testing(refer_dataset, 'val', parameters, batch_size)
                print('Average validation loss:{}'.format(self.total_loss[epoch]))

        return self.total_loss

    def run_testing(self, refer_dataset, split=None, parameters=None, batch_size=4):
        self.eval()
        refer_dataset.active_split = split
        dataloader = DataLoader(refer_dataset, batch_size=batch_size)

        total_loss = 0
        for k, instance in enumerate(tqdm(dataloader, desc='Validation')):
            with torch.no_grad():
                instances, targets = self.trim_batch(instance)
                self.clear_gradients(batch_size=targets.size()[0])

                label_scores = self(instances, parameters)
                total_loss += self.loss_function(label_scores, targets)
        return total_loss/float(k)

    def trim_batch(self, instance):
        pass

    def run_generate(self, refer_dataset, split=None, parameters=None):
        refer_dataset.active_split = split
        n = len(refer_dataset)
        dataloader = DataLoader(refer_dataset)

        generated_exp = [0]*len(refer_dataset)
        for k, instance in enumerate(tqdm(dataloader, desc='Generation')):
            generated_exp[k] = self.generate("<bos>", instance)

        return generated_exp

    def generate(self, start_word, instance):
        pass

    def clear_gradients(self, batch_size=None):
        self.zero_grad()

class SequenceLoss(nn.Module):
    def __init__(self, loss_function, disable_cuda=False):
        super(SequenceLoss, self).__init__()
        self.Loss = loss_function

        if not disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, embeddings, targets):
        loss = 0.0
        for step in range(targets.size()[1]):
            loss += self.Loss(embeddings[:, step, :], targets[:, step])

        return loss