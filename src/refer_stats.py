import numpy as np
import argparse, csv
from collections import defaultdict
from data_management.refer import REFER
from config import cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate referring expression for target object given bounding box and image')
    parser.add_argument('config_file', help='config file path')

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    refer = REFER(cfg)

    with open('locative_prep.txt', 'r') as f:
        locationPrep = list(set([l.strip() for l in f.readlines() if not l.startswith("#")]))

    locationPrep = [" {} ".format(l) for l in locationPrep if len(l)>0]
    locationPrep.sort(key=lambda item: (-len(item), item)) #Longest to shortest

    locCount1 = defaultdict(int)

    words = 0.0
    sentences = []
    for idx, sent in refer.Sents.items():
        words += len(sent['tokens'])
        sentences.append(" " + sent['sent'] + " ")


    for l in locationPrep:
        for i, sent in enumerate(sentences):
            if l in sent:
                locCount1[l] += 1
                sentences[i] = sent.replace(l, '')
    print("Average words:{}".format(words/len(refer.Sents)))
    total = np.sum(np.array(list(locCount1.values())))
    print("Total preps:{}".format(total))
    print("Average preps/sent{}".format(total/len(refer.Sents)))

    with open('{}_loc_freq.csv'.format(cfg.DATASET.NAME), 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for key, value in locCount1.items():
            spamwriter.writerow([key, value])

