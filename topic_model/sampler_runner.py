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
    p.add_argument('-p', type=int, help="num personas.")
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
                doc = json.loads(doc_str)
                docs.append(doc['paragraphs'])

    vocab_fp = os.path.join(args.i, 'vocab.txt')
    vocab = open(vocab_fp).read().split('\n')

    sampler = sampler_cy.BOW_Paragraph_GibbsSampler(docs=docs[:20000], vocab=vocab)

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
