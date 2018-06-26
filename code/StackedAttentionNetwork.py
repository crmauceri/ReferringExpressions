import os, pickle, argparse
from tqdm import *

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms

torch.manual_seed(1)

from DataLoader import DataLoader
from DataLoader import Instance
from AttentionModule import AttentionModule
from TruncatedImageNetworks import TruncatedVGGorAlex
from ClassifierHelper import Classifier

#Network Definition

class SAN(Classifier):

    def __init__(self, word_embedding, lstm_dim, hidden_dim, label_dim, use_cuda=False):
        super(SAN, self).__init__()
        self.lstm_dim = lstm_dim
        self.hidden_dim = hidden_dim
        self.label_dim = label_dim
        self.use_cuda = use_cuda

        self.word_embeddings = word_embedding
        self.embedding_dim = word_embedding.weight.size()[1]

        #Text Embedding Network
        self.lstm = nn.LSTM(self.embedding_dim +1, lstm_dim)
        self.hidden_lstm_state = self.init_hidden()

        #Image Embedding Network
        self.imagenet = TruncatedVGGorAlex(models.vgg16(pretrained=True))
        self.transform = transforms.Compose([
            transforms.Scale(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                  std = [0.229, 0.224, 0.225])
        ])
        self.image_dim = self.imagenet.output_dim
        self.imageTransformLayer = nn.Conv2d(self.image_dim[0], self.lstm_dim, 1)

        #Attention Module to merge text and image
        self.attend1 = AttentionModule(k=self.hidden_dim, d=self.lstm_dim, m=self.image_dim[1]*self.image_dim[2], use_cuda=use_cuda)
        self.attend2 = AttentionModule(k=self.hidden_dim, d=self.lstm_dim, m=self.image_dim[1]*self.image_dim[2], use_cuda=use_cuda)

        self.outLayer = nn.Sequential(nn.Linear(self.lstm_dim, self.label_dim), nn.LogSoftmax())

        self.total_loss = []
        self.start_epoch = 0

        if use_cuda:
            self.cuda()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state for the lstm.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.lstm_dim)),
                autograd.Variable(torch.zeros(1, 1, self.lstm_dim)))

    def forward(self, instance, parameters):
        image, l = parameters['images'][instance.image_id]
        image = autograd.Variable(image)
        image = image.unsqueeze(0)

        sentence = instance.inputs
        query_flags = instance.flags

        if self.use_cuda:
            sentence = sentence.cuda()
            query_flags = query_flags.cuda()
            image = image.cuda()

        #Language Layers
        embeds = self.word_embeddings(sentence)
        x = torch.cat([embeds, query_flags], 1)

        token_representation = x.view(len(sentence), 1, -1)
        lstm_out, self.hidden_lstm_state = self.lstm(token_representation)

        #Image Layers
        image_out = self.imagenet(image)
        image_out = self.imageTransformLayer(image_out)

        #Attention!
        attend1_out, attend1_weights = self.attend1(lstm_out[-1, :, :].view(self.lstm_dim, -1), image_out.view(self.lstm_dim, -1))
        attend2_out, attend2_weights = self.attend2(attend1_out, image_out.view(self.lstm_dim, -1))

        label_scores = self.outLayer(attend2_out.transpose(0,1))
        return label_scores

    def train(self, n_epochs, instances, images, checkpt_file):
        parameters = {'images': images}
        return super(SAN, self).run_training(n_epochs, instances, checkpt_file, parameters)
    #
    # def make_prediction(self, prepared_data):
    #     predictions = []
    #     for i in tqdm(range(len(prepared_data))):
    #         tag_scores = self(prepared_data[i][0], prepared_data[i][2], prepared_data[i][3])
    #         val, index = tag_scores[-1, :].data.max(0)
    #         predictions.append((index[0], val[0]))
    #     return predictions
    #
    # def load_model(self, checkpt_file):
    #     print("=> loading checkpoint '{}'".format(checkpt_file))
    #     checkpoint = torch.load(checkpt_file, map_location=lambda storage, loc: storage)
    #
    #     self.start_epoch = checkpoint['epoch']
    #     self.total_loss = checkpoint['total_loss']
    #     self.load_state_dict(checkpoint['state_dict'])
    #
    #     print("=> loaded checkpoint '{}' (epoch {})"
    #           .format(checkpt_file, checkpoint['epoch']))
    #
    # def train(self, n_epochs, instances, images, checkpt_file):
    #     loss_function = nn.NLLLoss()
    #     optimizer = optim.SGD(self.parameters(), lr=0.1)
    #
    #     if os.path.exists(checkpt_file) and os.path.isfile(checkpt_file):
    #         self.load_model(checkpt_file)
    #
    #     indices = list(range(len(instances)))
    #
    #     for epoch in range(self.start_epoch, n_epochs):
    #         # Shuffle examples in each batch
    #         #random.seed(1) #DEBUGGING
    #         #random.shuffle(indices)
    #         self.total_loss.append(0)
    #
    #         for j in tqdm(indices, desc='{}rd epoch'.format(epoch)):
    #
    #             instance = instances[j]
    #             if hasattr(instance, 'inputs'):
    #                 image, l = images[instance.image_id]
    #             else:
    #                 continue
    #             #image = autograd.Variable(image)
    #
    #             # Step 1. Remember that Pytorch accumulates gradients.  We need to clear them out
    #             # before each instance
    #             self.zero_grad()
    #
    #             # Also, we need to clear out the hidden state of the LSTM, detaching it from its
    #             # history on the last instance.
    #             self.hidden_lstm_state = self.init_hidden()
    #
    #             # Step 3. Run our forward pass.
    #             label_scores = self(instance.inputs, instance.flags, image)
    #
    #             # Step 4. Compute the loss, gradients, and update the parameters by calling
    #             # optimizer.step()
    #             if self.use_cuda:
    #                 loss = loss_function(label_scores.cuda(), instance.targets.cuda())
    #             else:
    #                 loss = loss_function(label_scores, instance.targets)
    #
    #             loss.backward()
    #             optimizer.step()
    #
    #             #if self.imagenet.state_dict()
    #
    #             self.total_loss[epoch] += loss.data[0]
    #
    #         torch.save({
    #             'epoch': epoch + 1,
    #             'state_dict': self.state_dict(),
    #             'total_loss': self.total_loss,
    #         }, checkpt_file)
    #
    #     return self.total_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Classify missing words with LSTM.')
    parser.add_argument('data_file', help='Pickle file generated by DataLoader')
    parser.add_argument('image_folder', help='Location of images')
    parser.add_argument('checkpoint_file', help='Filepath to save/load checkpoint. If file exists, checkpoint will be loaded')
    parser.add_argument('--epochs', dest='epochs', type=int, default=1,
                        help='Number of epochs to train (Default: 1)')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=100,
                        help='Size of LSTM embedding (Default:100)')
    parser.add_argument('--use_tokens', dest='use_tokens', type=bool, default=True,
                        help='If false, ignores pos token features. (Default:True)')
    parser.add_argument('--tag_dim', dest='tag_dim', type=int, default=10,
                        help='Size of tag embedding. If <1, will use one-hot representation (Default:10)')
    args = parser.parse_args()

    with open(args.data_file, 'rb') as f:
        data = pickle.load(f)

    use_cuda = torch.cuda.is_available()
    model = SAN(data.embed, args.hidden_dim, args.hidden_dim, len(data.labels), use_cuda)
    image_dataset = dset.ImageFolder(root=args.image_folder, transform=model.transform)
    print("Start Training")

    total_loss = model.train(args.epochs, data.instances, image_dataset, args.checkpoint_file)