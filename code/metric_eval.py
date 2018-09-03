from refer_python3.evaluation.refEvaluation import RefEvaluation
from refer_python3.refer import REFER
from csv import DictReader

# load refer of dataset
dataset = 'refcoco'
refer = REFER(dataset, splitBy = 'google')

# load generation outputs
with open('output/maoetal_baseline_batch_hidden1024_feats2005_dropout0.0_l21.0e-05.mdl_refcocog_15_generated.csv', newline='') as csvfile:
     genData = DictReader(csvfile)

# evaluate some refer expressions
refEval = RefEvaluation(refer, genData)
refEval.evaluate()

# print output evaluation scores
for metric, score in refEval.eval.items():
    print('%s: %.3f'%(metric, score))