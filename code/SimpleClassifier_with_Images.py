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

from SimpleClassifier_with_Tokens import LSTM
from ClassifierHelper import Classifier
from TruncatedImageNetworks import TruncatedVGGorAlex

#Network Definition
class LSTMPlusImageClassifier(Classifier):

    def __init__(self, word_embedding, tag_dim, lstm_dim, hidden_dim,
        label_dim, tag_embedding_dim=10, use_tokens=True, use_cuda=False):

        super(LSTMPlusImageClassifier, self).__init__()

        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda

        #Text Embedding Network
        self.wordnet = LSTM(word_embedding, tag_dim, hidden_dim, lstm_dim,
                                      tag_embedding_dim, use_tokens, use_cuda)

        #Image Embedding Network
        self.imagenet = TruncatedVGGorAlex(models.vgg16(pretrained=True), maxpool=True)
        self.transform = transforms.Compose([
            transforms.Scale(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                  std = [0.229, 0.224, 0.225])
        ])

        #Sequential Network to merge text and image
        self.seq_model = torch.nn.Sequential(
            torch.nn.Linear(self.imagenet.output_dim[0]*self.imagenet.output_dim[1]*self.imagenet.output_dim[2] + self.wordnet.output_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.label_dim),
            torch.nn.LogSoftmax()
        )

        self.total_loss = []
        self.start_epoch = 0

        if self.use_cuda:
            self.cuda()

    def forward(self, instance, parameters):
        image, l = parameters['images'][instance.image_id]
        image = autograd.Variable(image)
        image = image.unsqueeze(0)

        if self.use_cuda:
            image = image.cuda()

        #Text embedding
        lstm_out, self.hidden_lstm_state = self.wordnet(instance.inputs, instance.flags, instance.pos_variable)

        #Image embedding
        image_out = self.imagenet(image)

        #Concatenate image and text representations
        merge_image_text = torch.cat([lstm_out[-1].view(-1), image_out.view(-1)], 0)
        label_scores = self.seq_model(merge_image_text)
        return label_scores.view(-1, self.label_dim)

    def train(self, n_epochs, instances, images, checkpt_file):
        parameters = {'images':images}
        return super(LSTMPlusImageClassifier, self).train(n_epochs, instances, checkpt_file, parameters)


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
    model = LSTMPlusImageClassifier(data.embed, len(data.tags_to_idx), args.hidden_dim, args.hidden_dim, len(data.labels),
                           tag_embedding_dim=args.tag_dim, use_tokens=args.use_tokens, use_cuda=use_cuda)
    image_dataset = dset.ImageFolder(root=args.image_folder, transform=model.transform)

    print("Start Training")
    total_loss = model.train(args.epochs, data.instances, image_dataset, args.checkpoint_file)