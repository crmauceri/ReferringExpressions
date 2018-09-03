from MaoEtAl_baseline import LanguagePlusImage
import matplotlib.pyplot as plt

plt.figure(1, figsize=(9, 9))

with open('vocab_file.txt', 'r') as f:
    vocab = f.read().split()
# Add the start and end tokens
vocab.extend(['<bos>', '<eos>', '<unk>'])

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('models/') if isfile(join('models/', f)) and f.endswith('.mdl') and "batch" in f]

for checkpt_file in onlyfiles:
    model = LanguagePlusImage(checkpt_file=join('models/', checkpt_file), vocab=vocab)
    plt.plot(range(len(model.val_loss)), model.val_loss, label=checkpt_file)

leg = plt.legend(loc='best')

plt.show()