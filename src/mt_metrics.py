import argparse
import json

from nlgeval import NLGEval

from data_management.refer import REFER
from config import cfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculates metrics from output of a Generation/Comprehension network.' +
                                                 ' Run `run_network.py <config> test` first.')
    parser.add_argument('config_file', help='config file path')
    parser.add_argument('results_file', help='results file path')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    refer = REFER(cfg)

    #Self eval
    hypothesis = []
    references = []

    SELF_EVAL = False
    mp1 = 0.0
    mp2 = 0.0
    mean_objects = 0.0
    total = 0.0

    if SELF_EVAL:
        for ref_id, ref in refer.Refs.items():
            if len(ref['sentences'])>1:
                hypothesis.append(ref['sentences'][0]['sent'])
                references.append([s['sent'] for s in ref['sentences'][1:]])
    else:
        # load generation outputs
        with open(args.results_file, 'r') as f:
            genData = json.load(f)
            for row in genData:
                ref_id = int(row['refID'])
                gen_sentence = row['gen_sentence']
                hypothesis.append(row['gen_sentence'])
                references.append([s['sent'] for s in refer.Refs[ref_id]['sentences']])

                total += 1.0
                mean_objects += row['n_objects']
                mp1 += row['p@1']
                mp2 += row['p@2']

    references = list(zip(*references))
    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=['METEOR'])  # loads the models
    metrics_dict = nlgeval.compute_metrics(references, hypothesis)

    print("Bleu: %3.3f" % metrics_dict['Bleu_1'])
    print("ROUGE_L: %3.3f" % metrics_dict['ROUGE_L'])
    print("CIDEr: %3.3f" % metrics_dict['CIDEr'])

    print("MeanP@1: %3.3f" % (mp1/total))
    print("MeanP@2: %3.3f" % (mp2/total))
    print("Mean number of objects: %3.3f" % (mean_objects/total))
