from tqdm import tqdm as tqdm
import os
import sys
import sampler_cy
import json
from sklearn.model_selection import KFold
import numpy as np 

if __name__=='__main__':

    ## data
    data_input = []
    with open('labeled_data/doc_vecs.json') as data:
        for line in data:
            data_input.append(json.loads(line))
    vocab = open('labeled_data/vocab.txt').read().split('\n')[:-1]

    kf = KFold(n_splits=5)
    labeled_data_idx = list(filter(lambda x: x[1]['has_labels'], list(enumerate(data_input))))
    labeled_data_idx = np.array(list(map(lambda x: x[0], labeled_data_idx)))

    num_iterations = 1000
    kfold = 0

    for train_idx, test_idx in kf.split(labeled_data_idx):
        outdir = os.path.join('crossfold-topic-model', 'fold-%d' % kfold)
        os.makedirs(outdir, exist_ok=True)
        ## 
        test_examples = labeled_data_idx[test_idx]
        for idx in test_examples:
            data_input[idx]['has_labels'] = False

        ## sampler
        sampler = sampler_cy.BOW_Paragraph_GibbsSampler(data_input, vocab=vocab)
        sampler.initialize()
        for i in tqdm(range(num_iterations)):
            sampler.sample_pass()
        sampler.save_state(outdir)
        np.savetxt(os.path.join(outdir, 'test-examples.txt'), test_examples)    
        
        kfold += 1
        del sampler