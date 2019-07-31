from tqdm import *
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os

#torch.manual_seed(1)

DEBUG = False

class Classifier(nn.Module):
    def __init__(self, cfg, loss_function):
        super(Classifier, self).__init__()
        self.cfg = cfg
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

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpt_file, checkpoint['epoch']))

    def save_model(self, checkpt_prefix, params):
        print("=> saving '{}'".format(checkpt_prefix))
        torch.save(params, checkpt_prefix)

    @staticmethod
    def checkpt_file(cfg, epoch):
        return 'checkpoints/{}.mdl.checkpoint{}'.format(cfg.OUTPUT.CHECKPOINT_PREFIX, epoch)

    @staticmethod
    def model_file(cfg):
        return 'models/{}.mdl'.format(cfg.OUTPUT.CHECKPOINT_PREFIX)

    @staticmethod
    def test_output_file(cfg):
        return 'output/{}_{}_test.json'.format(cfg.OUTPUT.CHECKPOINT_PREFIX, cfg.DATASET.NAME)

    def run_training(self, refer_dataset):
        log_dir = os.path.join("output", self.cfg.OUTPUT.CHECKPOINT_PREFIX)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        # TODO currently throwing error (DepthVGGorAlex object argument after * must be an iterable, not NoneType)
        # writer.add_graph(self.cpu(), images)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                               lr=self.cfg.TRAINING.LEARNING_RATE, weight_decay=self.cfg.TRAINING.L2_FRACTION)

        if isinstance(refer_dataset, tuple):
            train_dataset = refer_dataset[0]
            test_dataset = refer_dataset[1]
            val_dataset = refer_dataset[2]
        else:
            train_dataset = refer_dataset
            test_dataset = refer_dataset
            val_dataset = refer_dataset

        train_dataset.active_split = 'train'

        if self.use_cuda:
            dataloader = DataLoader(train_dataset, self.cfg.TRAINING.BATCH_SIZE, shuffle=True)
        else:
            dataloader = DataLoader(train_dataset, self.cfg.TRAINING.BATCH_SIZE, shuffle=True, num_workers=4)

        print("Before training")
        train_loss = self.compute_average_loss(train_dataset, 'train', batch_size=self.cfg.TRAINING.BATCH_SIZE)
        print('Average training loss:{}'.format(train_loss))
        self.display_metrics(train_dataset, 'train')

        for epoch in range(self.start_epoch, self.cfg.TRAINING.N_EPOCH):
            self.train()
            train_dataset.active_split = 'train'
            self.total_loss.append(0)

            for i_batch, sample_batched in enumerate(tqdm(dataloader, desc='{}rd epoch'.format(epoch))):

                instances, targets = self.trim_batch(sample_batched)
                self.clear_gradients(batch_size=targets.size()[0])

                label_scores = self(instances)
                loss = self.loss_function(label_scores, targets)
                loss.backward()
                optimizer.step()

                self.total_loss[epoch] += loss.item()

                if DEBUG and i_batch == 5:
                    break


            self.total_loss[epoch] = self.total_loss[epoch] / float(i_batch + 1)
            writer.add_scalar('Average training loss', self.total_loss[epoch], global_step=epoch)

            self.save_model(Classifier.model_file(self.cfg), {
                'epoch': epoch + 1,
                'state_dict': self.state_dict(),
                'total_loss': self.total_loss,
                'val_loss': self.val_loss})

            print('Average training loss:{}'.format(self.total_loss[epoch]))

            if epoch % self.cfg.TRAINING.VALIDATION_FREQ == 0:
                # Log weights and gradients
                for tag, value in self.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram(tag, value.data.cpu().numpy(), epoch)
                    # writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

                # Testing the training set can be very time consuming. Off by default.
                if self.cfg.TEST.DO_TRAIN:
                    self.display_metrics(train_dataset, 'train', writer=writer, epoch=epoch)

                # Validation set is usually smaller. On by default.
                if self.cfg.TEST.DO_VAL:
                    val_loss = self.compute_average_loss(val_dataset, 'val', batch_size=self.cfg.TRAINING.BATCH_SIZE)
                    writer.add_scalar('Average validation loss', val_loss, global_step=epoch)
                    self.val_loss.append(val_loss)

                    print('Average validation loss:{}'.format(self.val_loss[-1]))
                    self.display_metrics(val_dataset, 'val', writer=writer, epoch=epoch)

                # Save checkpoint
                self.save_model(Classifier.checkpt_file(self.cfg, epoch), {
                    'epoch': epoch,
                    'state_dict': self.state_dict(),
                    'total_loss': self.total_loss,
                    'val_loss': self.val_loss})

        writer.close()
        return self.total_loss

    def compute_average_loss(self, refer_dataset, split=None, batch_size=4):
        self.eval()
        refer_dataset.active_split = split
        dataloader = DataLoader(refer_dataset, batch_size=batch_size)

        total_loss = 0
        for k, instance in enumerate(tqdm(dataloader, desc='Average loss on {}'.format(split))):
            with torch.no_grad():
                instances, targets = self.trim_batch(instance)
                self.clear_gradients(batch_size=targets.size()[0])

                label_scores = self(instances)
                total_loss += self.loss_function(label_scores, targets)

            if DEBUG and k == 5:
                break

        return total_loss/float(k)

    def run_test(self, refer_dataset, split=None):
        self.eval()
        refer_dataset.active_split = split

        # This is a hack to make sure that comprehension works in MaoEtAl_baseline regardless of whether contrast objects were used in training
        if hasattr(refer_dataset, 'n_contrast_object'):
            refer_dataset.n_contrast_object = float('inf')

        dataloader = DataLoader(refer_dataset, batch_size=1, shuffle=True)

        output = list()
        for k, batch in enumerate(tqdm(dataloader, desc='Test {}'.format(split))):
            instances, targets = self.trim_batch(batch)
            output.append(self.test(instances, targets))

            # Large test sets can be very slow to process. Therefore, default only processes a random sample of 10000
            if not self.cfg.TEST.DO_ALL and k > 10000:
                break

            if DEBUG and k == 5:
                break

        return output

    def display_metrics(self, refer_dataset, split=None, verbose=False, writer=None, epoch=0):
        output = self.run_test(refer_dataset, split)
        metrics = self.run_metrics(output, refer_dataset)

        for key, value in metrics.items():
            if isinstance(value, list) and verbose:
                headers = value[0].keys()
                print("\t".join(headers))
                for entry in value:
                    print("\t".join(entry.values()))
            elif not isinstance(value, list):
                print('{}:\t{:.3f}'.format(key, value))
                if writer is not None:
                    writer.add_scalar('{}_{}'.format(split, key), value, global_step=epoch)

    def run_metrics(self, output, refer):
        pass

    def test(self, instance, targets):
        pass

    def trim_batch(self, instance):
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
            reduction_setting = self.Loss.reduction
            # If you use the per_instance setting, your Loss function must have reduction=='none'
            self.Loss.reduction = 'none'
            loss = torch.zeros(embeddings.size()[0], device=self.device)
            for step in range(targets.size()[1]):
                #TODO try the mean here instead
                loss += self.Loss(embeddings[:, step, :], targets[:, step])
            self.Loss.reduction = reduction_setting
        else:
            loss = 0.0
            for step in range(targets.size()[1]):
                loss += self.Loss(embeddings[:, step, :], targets[:, step])

        return loss