from tqdm import tqdm
from refer import REFER
import numpy as np
from csv import DictReader

from nlg_eval import NLGEval

refer = REFER(dataset='refcocog', splitBy ='google', data_root='pyutils/refer_python3/data')

#Self eval
hypothesis = []
references = []

SELF_EVAL = False

if SELF_EVAL:
    for ref_id, ref in refer.Refs.items():
        if len(ref['sentences'])>1:
            hypothesis.append(ref['sentences'][0]['sent'])
            references.append([s['sent'] for s in ref['sentences'][1:]])
else:
    # load generation outputs
    with open('/Users/Mauceri/Workspace/ReferExpGeneration/output/maoetal_finetune60_lr1e-5_hidden1024_feats2005_dropout0.0_l21.0e-05.mdl_refcocog_90_generated.csv',
              newline='') as csvfile:
        genData = DictReader(csvfile)
        for row in genData:
            ref_id = int(row['refID'])
            gen_sentence = row['generated_sentence'].replace('<eos>', '')
            hypothesis.append(gen_sentence)
            references.append([s['sent'] for s in refer.Refs[ref_id]['sentences']])


nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=['METEOR'])  # loads the models
metrics_dict = nlgeval.compute_metrics(references, hypothesis)

print("Bleu: %3.3f" % metrics_dict['Bleu_1'])
print("ROUGE_L: %3.3f" % metrics_dict['ROUGE_L'])
print("CIDEr: %3.3f" % metrics_dict['CIDEr'])
