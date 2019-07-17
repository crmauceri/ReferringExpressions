from tqdm import tqdm
import numpy as np
import argparse

from .data_management.refer import REFER
from .config import cfg

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

    locationPrep = [l for l in locationPrep if len(l)>0]
    loc_index = dict(zip(locationPrep, range(len(locationPrep))))
    locCount1 = [0] * len(locationPrep)


    words = 0.0
    for idx, sent in tqdm(refer.Sents.items()):
        words += len(sent['tokens'])
        for l in locationPrep:
            if l in sent['tokens']:
                locCount1[loc_index[l]] += 1

    print("Average words:{}".format(words/len(refer.Sents)))
    print("Average preps/sent{}".format(np.sum(np.array(locCount1))/len(refer.Sents)))

    sorted_idx = np.argsort(np.array(locCount1))
    print("Most frequent spatial preps:{}".format(" ".join([locationPrep[ii] for ii in sorted_idx[-1:-10:-1]])))