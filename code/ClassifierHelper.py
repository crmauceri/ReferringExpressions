import random, os
from tqdm import *

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

class Classifier(nn.Module):
    def __init__(self, use_cuda=False):
        super(Classifier, self).__init__()
        self.use_cuda = use_cuda
        self.total_loss = []
        self.start_epoch = 0

    def forward(self, instance, parameters):
        pass

    def load_model(self, checkpt_file):
        print("=> loading checkpoint '{}'".format(checkpt_file))
        checkpoint = torch.load(checkpt_file, map_location=lambda storage, loc: storage)

        self.start_epoch = checkpoint['epoch']
        self.total_loss = checkpoint['total_loss']
        self.load_state_dict(checkpoint['state_dict'])
        self.load_params(checkpoint)

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpt_file, checkpoint['epoch']))

    def load_params(self, checkpoint):
        pass

    def save_model(self, checkpt_file, params):
        print("=> saving checkpoint '{}'".format(checkpt_file))
        torch.save(params, checkpt_file)

    def train(self, n_epochs, instances, checkpt_file, parameters=None, debug=False):
        loss_function = nn.NLLLoss()
        if self.use_cuda:
            loss_function = loss_function.cuda()

        optimizer = optim.SGD(self.parameters(), lr=0.1)

        if os.path.exists(checkpt_file) and os.path.isfile(checkpt_file):
            self.load_model(checkpt_file)

        indices = list(range(len(instances)))

        for epoch in range(self.start_epoch, n_epochs):
            # Shuffle examples in each batch
            if debug:
                random.seed(1) #DEBUGGING
            random.shuffle(indices)
            self.total_loss.append(0)

            for j in tqdm(indices, desc='{}rd epoch'.format(epoch)):

                instance = instances[j]

                self.clear_gradients()

                label_scores = self(instance, parameters)
                targets = self.targets(instance)

                if self.use_cuda:
                    loss = loss_function(label_scores.cuda(), targets.cuda())
                else:
                    loss = loss_function(label_scores, targets)

                loss.backward()
                optimizer.step()

                self.total_loss[epoch] += loss.item()

            self.save_model(checkpt_file, {
                'epoch': epoch + 1,
                'state_dict': self.state_dict(),
                'total_loss': self.total_loss})

            print('Average loss:{}'.format(self.total_loss[epoch]/float(len(instances))))

        return self.total_loss

    def targets(self, instance):
        pass

    def clear_gradients(self):
        self.zero_grad()

    def make_prediction(self, instances, parameters):
        predictions = []
        with torch.no_grad():
            for i in tqdm(range(len(instances))):
                instance = instances[i]
                tag_scores = self(instance, parameters)
                val, index = tag_scores.data.max(0)
                predictions.append((index[0], val[0]))
        return predictions
