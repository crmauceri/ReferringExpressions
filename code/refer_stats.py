from tqdm import tqdm
from refer_python3.refer import REFER
import numpy as np

with open('/Users/Mauceri/Workspace/SUNSpotCollection/locative_prep.txt', 'r') as f:
    locationPrep = list(set([l.strip() for l in f.readlines() if not l.startswith("#")]))

locationPrep = [l for l in locationPrep if len(l)>0]
loc_index = dict(zip(locationPrep, range(len(locationPrep))))
locCount1 = [0] * len(locationPrep)

refer = REFER(dataset='refcocog', splitBy ='google', data_root='pyutils/refer_python3/data')
#refer_sunspot = REFER(dataset='sunspot', splitBy = 'boulder', data_root='pyutils/refer_python3/data')

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

refer = REFER(dataset='sunspot', splitBy = 'boulder', data_root='pyutils/refer_python3/data')
#refer_sunspot = REFER(dataset='sunspot', splitBy = 'boulder', data_root='pyutils/refer_python3/data')

locCount2 = [0] * len(locationPrep)

words = 0.0
for idx, sent in tqdm(refer.Sents.items()):
    words += len(sent['tokens'])
    for l in locationPrep:
        if l in sent['tokens']:
            locCount2[loc_index[l]] += 1

print("Average words:{}".format(words/len(refer.Sents)))
print("Average preps/sent:{}".format(np.sum(np.array(locCount2))/len(refer.Sents)))

sorted_idx = np.argsort(np.array(locCount2))
print("Most frequent spatial preps:{}".format(" ".join([locationPrep[ii] for ii in sorted_idx[-1:-10:-1]])))