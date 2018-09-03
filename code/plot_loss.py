from MaoEtAl_baseline import LanguagePlusImage
import matplotlib.pyplot as plt

ax = plt.subplot(111)

with open('vocab_file.txt', 'r') as f:
    vocab = f.read().split()
# Add the start and end tokens
vocab.extend(['<bos>', '<eos>', '<unk>'])

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('models/') if isfile(join('models/', f)) and f.endswith('.mdl')]

for checkpt_file in onlyfiles:
    model = LanguagePlusImage(checkpt_file=join('models/', checkpt_file), vocab=vocab)
    plt.plot(range(len(model.total_loss)), model.total_loss, label=checkpt_file)

leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)

plt.show()