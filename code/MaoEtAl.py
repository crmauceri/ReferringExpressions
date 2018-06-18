import argparse, os

import torch
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms

torch.manual_seed(1)

from TruncatedImageNetworks import TruncatedVGGorAlex
from LSTM import LanguageModel, find_vocab
from ClassifierHelper import Classifier

from refer_python3.refer import REFER

#Network Definition
class LanguagePlusImage(Classifier):

    def __init__(self, checkpt_file=None, word_embedding=None, hidden_dim=None, use_cuda=False):
        super(LanguagePlusImage, self).__init__(use_cuda)

        #Text Embedding Network
        self.wordnet = LanguageModel(checkpt_file=checkpt_file, additional_feat=2005, word_embedding=word_embedding,
                                     hidden_dim=hidden_dim, use_cuda=use_cuda)

        #Image Embedding Network
        self.imagenet = TruncatedVGGorAlex(models.vgg16(pretrained=True), maxpool=True)
        self.transform = transforms.Compose([
            transforms.Scale(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                  std = [0.229, 0.224, 0.225])
        ])

        self.to(self.device)
        if checkpt_file is not None:
            super(LanguagePlusImage, self).load_model(checkpt_file)


    def forward(self, instance, parameters):
        #Global feature
        image, l = parameters['images'][instance.image_id]
        image = image.unsqueeze(0)
        if self.use_cuda:
            image = image.cuda()
        image_out = self.imagenet(image)

        #Object feature
        object, l = parameters['object'][instance.image_id]
        object = object.unsqueeze(0)
        if self.use_cuda:
            object = object.cuda()
        object_out = self.imagenet(object)

        #Position features
        #[top_left_x / W, top_left_y/H, bottom_left_x/W, bottom_left_y/H, size_bbox/size_image]
        pos = torch.Tensor([], dtype=torch.float, device=self.device, requires_grad=True)

        #Concatenate image representations
        instance['feats'] = torch.cat([image_out, object_out, pos], 1)

        #Input to LanguageModel
        return self.wordnet(ref=instance)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Classify missing words with LSTM.')
    parser.add_argument('mode', help='train/test')
    parser.add_argument('checkpoint_file',
                        help='Filepath to save/load checkpoint. If file exists, checkpoint will be loaded')

    parser.add_argument('--data_root', help='path to data directory', default='pyutils/refer_python3/data')
    parser.add_argument('--dataset', help='dataset name', default='refcocog')
    parser.add_argument('--splitBy', help='team that made the dataset splits', default='google')
    parser.add_argument('--epochs', dest='epochs', type=int, default=1,
                        help='Number of epochs to train (Default: 1)')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=1024,
                        help='Size of LSTM embedding (Default:100)')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0, help='Dropout probability')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    refer = REFER(args.data_root, args.dataset, args.splitBy)
    vocab = find_vocab(refer)
    # Add the start and end tokens
    vocab.extend(['<bos>', '<eos>', '<unk>'])

    if (os.path.isfile(args.checkpoint_file)):
        model = LanguagePlusImage(checkpt_file=args.checkpoint_file, use_cuda=use_cuda)
    else:
        model = LanguagePlusImage(vocab=vocab, hidden_dim=args.hidden_dim,
                              use_cuda=use_cuda, dropout=args.dropout)

    image_dataset = dset.ImageFolder(root=args.image_folder)

    # Preprocess REFER dataset
    model.sent2vocab(refer)

    if (args.mode == 'train'):
        print("Start Training")
        total_loss = model.train(args.epochs, refer.loadSents(refer.getRefIds(split='train')), args.checkpoint_file)

    if (args.mode == 'test'):
        print("Start Testing")
        print(model.generate([]))

