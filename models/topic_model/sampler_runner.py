from tqdm import tqdm
import pickle
import glob, re
import random
import os
import json
import sampler_cy


if __name__=="__main__":
    import argparse; p=argparse.ArgumentParser()
    # model params
    p.add_argument('-i', type=str, help="input directory.")
    p.add_argument('-o', type=str, help="output directory.")
    p.add_argument('-k', type=int, help="num topics.")
    p.add_argument('-p', type=int, help="num paragraph types.")
    p.add_argument('-t', type=int, help="num iterations.")
    p.add_argument('--use-cached', default=False, action='store_true', dest='use_cached', help='use intermediate cached file.')
    args = p.parse_args()

    here = os.path.dirname(__file__)
    input_documents_fp = os.path.join(here, args.i, 'doc_vecs.json')
    output_dir = os.path.join(here, args.o)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_documents_fp) as f:
        doc_strs =  f.read().split('\n')
        docs = []
        for doc_str in doc_strs:
            if doc_str:
                docs.append(json.loads(doc_str))

    vocab_fp = os.path.join(args.i, 'vocab.txt')
    vocab = open(vocab_fp).read().split('\n')
    if 'label' in args.i:
        num_partypes = len(set([l for doc in docs for l in doc['labels']])) - 1
    else:
        num_partypes = args.p

    print('Training with %d partypes...' % num_partypes)
    sampler = sampler_cy.BOW_Paragraph_GibbsSampler(docs=docs, vocab=vocab, num_partypes=num_partypes)

    ##
    cached_files = glob.glob(os.path.join(output_dir, 'trained-sampled-iter*'))
    if not args.use_cached or (len(cached_files) == 0):
        prev_iter = 0
        sampler.initialize()
    else:
        print('loading...')
        max_file = max(cached_files, key=lambda x: int(re.findall('iter-(\d+)', x)[0]))
        sampler = pickle.load(open(max_file, 'rb'))
        prev_iter = int(re.findall('iter-(\d+)', max_file)[0])

    i = 0
    for i in tqdm(range(args.t), total=args.t):
        if i % 100 == 1:
            model_outdir = os.path.join(output_dir, 'model-state-iter-%d.pkl' % i) 
            if not os.path.exists(model_outdir):
                os.makedirs(model_outdir)
            sampler.save_state(model_outdir)
        sampler.sample_pass()

    ## done
    model_outdir = os.path.join(output_dir, 'model-state-iter-%d.pkl' % i) 
    if not os.path.exists(model_outdir):
        os.makedirs(model_outdir)
    sampler.save_state(model_outdir)
