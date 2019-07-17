from tqdm import *

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

#torch.manual_seed(1)

DEBUG = False

class Classifier(nn.Module):
    def __init__(self, cfg, loss_function):
        super(Classifier, self).__init__()
        self.total_loss = []
        self.val_loss = []
        self.start_epoch = 0
        self.loss_function = loss_function

        if not cfg.MODEL.DISABLE_CUDA and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.use_cuda = True
            print("Using cuda")
        else:
            self.device = torch.device('cpu')
            self.use_cuda = False
            print("Using cpu")

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

    @staticmethod
    def checkpt_file(cfg, epoch):
        return 'checkpoints/{}.mdl.checkpoint{}'.format(cfg.OUTPUT.CHECKPOINT_PREFIX, epoch)

    @staticmethod
    def model_file(cfg):
        return 'models/{}.mdl'.format(cfg.OUTPUT.CHECKPOINT_PREFIX)

    @staticmethod
    def generated_output_file(cfg):
        return 'output/{}_{}_generated.csv'.format(cfg.OUTPUT.CHECKPOINT_PREFIX, cfg.DATASET.NAME)

    @staticmethod
    def comprehension_output_file(cfg):
        'output/{}_{}_comprehension.csv'.format(cfg.OUTPUT.CHECKPOINT_PREFIX, cfg.DATASET.NAME)

    def run_training(self, refer_dataset, cfg):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                               lr=cfg.TRAINING.LEARNING_RATE, weight_decay=cfg.TRAINING.L2_FRACTION)

        refer_dataset.active_split = 'train'

        if self.use_cuda:
            dataloader = DataLoader(refer_dataset, cfg.TRAINING.BATCH_SIZE, shuffle=True)
        else:
            dataloader = DataLoader(refer_dataset, cfg.TRAINING.BATCH_SIZE, shuffle=True, num_workers=4)

        for epoch in range(self.start_epoch, cfg.TRAINING.N_EPOCH):
            self.train()
            refer_dataset.active_split = 'train'
            self.total_loss.append(0)

            for i_batch, sample_batched in enumerate(tqdm(dataloader, desc='{}rd epoch'.format(epoch))):

                instances, targets = self.trim_batch(sample_batched)
                self.clear_gradients(batch_size=targets.size()[0])

                loss = 0
                label_scores = self(instances)
                loss += self.loss_function(label_scores, targets)

                if DEBUG:
                    print([self.wordnet.ind2word[instances['vocab_tensor'][0, i]] for i in range(instances['vocab_tensor'].size()[1])])
                    print([self.wordnet.ind2word[torch.argmax(label_scores[0, i, :])] for i in range(instances['vocab_tensor'].size()[1]-1)])
                    print(loss)

                loss.backward()
                optimizer.step()

                if DEBUG:
                    self.clear_gradients(batch_size=1)
                    print(self.generate('<bos>', feats=instances['feats'][0]))

                self.total_loss[epoch] += loss.item()

            self.total_loss[epoch] = self.total_loss[epoch] / float(i_batch)

            self.clear_gradients(batch_size=1)
            self.save_model(Classifier.model_file(cfg), {
                'epoch': epoch + 1,
                'state_dict': self.state_dict(),
                'total_loss': self.total_loss,
                'val_loss': self.val_loss})

            print('Average training loss:{}'.format(self.total_loss[epoch]))

            if epoch % 2 == 0:
                self.save_model(Classifier.checkpt_file(cfg, epoch), {
                    'epoch': epoch,
                    'state_dict': self.state_dict(),
                    'total_loss': self.total_loss,
                    'val_loss': self.val_loss})

                self.val_loss.append(0)
                self.val_loss[-1] = self.run_testing(refer_dataset, 'val', batch_size=cfg.TRAINING.BATCH_SIZE)
                print('Average validation loss:{}'.format(self.total_loss[epoch]))

        return self.total_loss

    def run_testing(self, refer_dataset, split=None, batch_size=4):
        self.eval()
        refer_dataset.active_split = split
        dataloader = DataLoader(refer_dataset, batch_size=batch_size)

        total_loss = 0
        for k, instance in enumerate(tqdm(dataloader, desc='Validation')):
            with torch.no_grad():
                instances, targets = self.trim_batch(instance)
                self.clear_gradients(batch_size=targets.size()[0])

                label_scores = self(instances)
                total_loss += self.loss_function(label_scores, targets)
        return total_loss/float(k)

    def trim_batch(self, instance):
        pass

    def run_generate(self, refer_dataset, split=None):
        refer_dataset.active_split = split
        n = len(refer_dataset)
        dataloader = DataLoader(refer_dataset)

        generated_exp = [0]*len(refer_dataset)
        for k, instance in enumerate(tqdm(dataloader, desc='Generation')):
            instances, targets = self.trim_batch(instance)
            generated_exp[k] = dict()
            generated_exp[k]['generated_sentence'] = ' '.join(self.generate("<bos>", instance=instances))
            generated_exp[k]['refID'] = instance['refID'].item()
            generated_exp[k]['imgID'] = instance['imageID'].item()
            generated_exp[k]['objID'] = instance['objectID'][0]
            generated_exp[k]['objClass'] = instance['objectClass'][0]

        return generated_exp

    def generate(self, start_word, instance=None, feats=None):
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

    def forward(self, embeddings, targets, per_instance=False):
        if per_instance:
            loss = torch.zeros(embeddings.size()[0], device=self.device)
            for step in range(targets.size()[1]):
                loss += self.Loss(embeddings[:, step, :], targets[:, step])
        else:
            loss = 0.0
            for step in range(targets.size()[1]):
                loss += self.Loss(embeddings[:, step, :], targets[:, step])

        return loss