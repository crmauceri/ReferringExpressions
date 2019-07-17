import argparse
from csv import DictReader

from nlg_eval import NLGEval

from data_management.refer import REFER
from config import cfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate referring expression for target object given bounding box and image')
    parser.add_argument('config_file', help='config file path')

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    refer = REFER(cfg)

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
        with open('{}_{}_{}_generated.csv'.format(model.checkpt_file().replace('models', 'output'),
                                                  cfg.DATASET.NAME, model.start_epoch),
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
