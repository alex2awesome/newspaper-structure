import numpy as np
import os
import json
from tqdm import tqdm
import pickle
import glob, re
import random

def categorical(k, p):
    return np.random.choice(range(k), p=p)

def bernoilli(p):
    return np.random.binomial(1, p)

class BOW_Paragraph_GibbsSampler():
    def __init__(
        self, docs, vocab, num_partypes=10, num_topics=25, 
        H_P_prior=1, H_T_prior=1, gamma=.7, alpha=1, beta=1
    ):
        # data
        self.docs = docs
        self.vocab = vocab
        self.n_docs = len(docs)
        self.n_graphs = sum(list(map(lambda x: len(x), docs)))
        self.n_vocab = len(vocab)

        # hyperparameters
        self.num_partypes = num_partypes
        self.num_topics = num_topics
        self.par_types = list(range(self.num_partypes))
        self.topics = list(range(self.num_topics))

        # priors
        self.H_T = H_T_prior
        self.H_P = H_P_prior
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

        # counts 
        ## doc-type probability
        #### I(doc d has type t)
        #### count_t_d
        self.partype_counts = np.zeros(self.num_partypes)
        self.pardoc_to_type = {}
        self.old_pardoc_to_type = {}

        # switching variable
        self.switching_variable = {}
        self.old_switching_variable = {}

        ## background word-topic probabilities
        #### I(doc d has type t, word w has topic k)
        #### count_k_d_t_w
        self.doc__wordtopic_counts = np.zeros(self.n_docs)
        self.doc_by_wordtopic__wordtopic_counts = np.zeros((self.n_docs, self.num_topics))

        ## paragraph word-topic probabilities
        #### I(source S_d in doc d has type s, word w has topic k)
        #### count_k_d_s_Sd_w
        self.partype__wordtopic_counts = np.zeros(self.num_partypes)
        self.partype_by_wordtopic__wordtopic_counts = np.zeros((self.num_partypes, self.num_topics))

        self.word_to_topic = {} 

        ## word probabilities
        self.vocab_by_wordtopic__word_counts = np.zeros((self.n_vocab, self.num_topics))
        self.wordtopic__word_counts = np.zeros(self.num_topics)


    def initialize(self):
        ## probability vectors
        partype_prob_vec = np.ones(self.num_partypes) / self.num_partypes
        z_prob_vec = np.ones(self.num_topics) / self.num_topics

        ## iterate through documents
        for doc_id, doc in tqdm(enumerate(self.docs), total=len(self.docs)):
            for par_id, par in enumerate(doc):
                partype = categorical(self.num_partypes, p=partype_prob_vec)
                self.partype_counts[partype] += 1
                self.pardoc_to_type[(doc_id, par_id)] = partype
                self.old_pardoc_to_type[(doc_id, par_id)] = partype

                for word_id, word in enumerate(par):
                    s = bernoilli(self.gamma)
                    self.switching_variable[(doc_id, par_id, word_id)] = s
                    self.old_switching_variable[(doc_id, par_id, word_id)] = s

                    word_topic = categorical(self.num_topics, p=z_prob_vec)
                    if s == 0:
                        ## set paragraph word-topics counts
                        self.partype_by_wordtopic__wordtopic_counts[partype, word_topic] += 1
                        self.partype__wordtopic_counts[partype] += 1
                        self.vocab_by_wordtopic__word_counts[word, word_topic] += 1
                        self.wordtopic__word_counts[word_topic] += 1

                    else:
                        ## set document word-topic counts
                        self.doc_by_wordtopic__wordtopic_counts[doc_id, word_topic] += 1
                        self.doc__wordtopic_counts[doc_id] += 1
                        self.vocab_by_wordtopic__word_counts[word, word_topic] += 1
                        self.wordtopic__word_counts[word_topic] += 1

                    ## cache
                    self.word_to_topic[(doc_id, par_id, word_id)] = word_topic

    ###
    # sample par type
    ###
    def partype_prob(self, proposed_partype, doc_id, par_id):
        ## partype factor
        partype_term = np.log(self.H_T + self.partype_counts[proposed_partype])

        ## paragraph word_topic factor
        paragraph_wordtopic_term = 0
        log_denom = np.log(self.num_topics * self.H_P + self.partype__wordtopic_counts[proposed_partype])
        for word_id, word in enumerate(self.docs[doc_id][par_id]):
            if paragraph_wordtopic_term < -100: ## hack... is there any other way to deal with very low probabilities?
                break
            if self.switching_variable[(doc_id, par_id, word_id)] == 0:
                wordtopic = self.word_to_topic[(doc_id, par_id, word_id)]
                num = self.H_P + self.partype_by_wordtopic__wordtopic_counts[proposed_partype, wordtopic]
                paragraph_wordtopic_term += (np.log(num) - log_denom)

        ## combine and return
        log_prob = partype_term + paragraph_wordtopic_term
        return np.exp(log_prob)

    def propose_new_partype(self, doc_id, par_id):
        par_prob_vec = np.zeros(self.num_partypes)
        for t in self.par_types:
            par_prob_vec[t] = self.partype_prob(t, doc_id, par_id)
        new_partype = categorical(self.num_partypes, p=par_prob_vec / par_prob_vec.sum())
        return new_partype

    def sample_partype(self):
        ## for each doc, decrement counts and resample 
        for doc_id, doc in tqdm(enumerate(self.docs), total=len(self.docs)):
            for par_id, par in enumerate(doc):
                old_partype = self.pardoc_to_type[(doc_id, par_id)]
                ## decrement
                self.partype_counts[old_partype] -= 1           
                new_partype = self.propose_new_partype(doc_id, par_id)
                ## increment
                self.partype_counts[new_partype] += 1
                ## cache
                self.pardoc_to_type[(doc_id, par_id)] = new_partype
                self.old_pardoc_to_type[(doc_id, par_id)] = old_partype


    ## 
    # Sample switching variable...
    ## 
    def switching_prob_0(self, partype, wordtopic):
        '''Gamma = 0 is paragraph word.'''
        wordtopic_term = (
            (self.partype_by_wordtopic__wordtopic_counts[partype, wordtopic] + self.H_P)
        / 
            (self.partype__wordtopic_counts[partype] + self.num_topics * self.H_P)
        )
        return self.gamma * wordtopic_term

    def switching_prob_1(self, doc_id, wordtopic):
        '''Gamma = 1 is document word.'''
        wordtopic_term = (
            (self.doc_by_wordtopic__wordtopic_counts[doc_id, wordtopic] + self.alpha)
        / 
            (self.doc__wordtopic_counts[doc_id] + self.num_topics * self.alpha)
        )
        return (1 - self.gamma) * wordtopic_term

    def sample_switching_variable(self, doc_id, par_id, word_id):
        partype = self.pardoc_to_type[(doc_id, par_id)]
        wordtopic = self.word_to_topic[(doc_id, par_id, word_id)]
        p_s_0 = self.switching_prob_0(partype, wordtopic)
        p_s_1 = self.switching_prob_1(doc_id, wordtopic)
        p = p_s_0 / (p_s_0 + p_s_1)
        new_switching_var = bernoilli(p)
        self.switching_variable[(doc_id, par_id, word_id)] = new_switching_var
        return new_switching_var

    ###
    # Sample word topic
    ###
    def doc_wordtopic_prob(self, proposed_wordtopic, doc_id, par_id, word_id):
        # old_wordtopic = self.word_to_backgroundtopic[(doc_id, par_id, word_id)]
        word = self.docs[doc_id][par_id][word_id]

        ## source word_topic factor
        wordtopic_term = self.alpha + self.doc_by_wordtopic__wordtopic_counts[doc_id, proposed_wordtopic]

        ## word factor
        denom = self.n_vocab * self.beta + self.wordtopic__word_counts[proposed_wordtopic]
        num = self.beta + self.vocab_by_wordtopic__word_counts[word, proposed_wordtopic]
        word_term = (num / denom)

        ## combine and return 
        return wordtopic_term * word_term 

    def propose_new_doc_wordtopic(self, doc_id, par_id, word_id):
        doc_wordtopic_prob_vec = np.zeros(self.num_topics)
        for k in self.topics:
            doc_wordtopic_prob_vec[k] = self.doc_wordtopic_prob(k, doc_id, par_id, word_id)
        wordtopic = categorical(self.num_topics, p=doc_wordtopic_prob_vec / doc_wordtopic_prob_vec.sum())
        return wordtopic


    def par_wordtopic_prob(self, proposed_wordtopic, doc_id, par_id, word_id):
        word = self.docs[doc_id][par_id][word_id]
        partype = self.pardoc_to_type[(doc_id, par_id)]

        ## source word_topic factor
        wordtopic_term = self.H_P + self.partype_by_wordtopic__wordtopic_counts[partype, proposed_wordtopic]

        ## word factor
        denom = self.n_vocab * self.beta + self.wordtopic__word_counts[proposed_wordtopic]
        num = self.beta + self.vocab_by_wordtopic__word_counts[word, proposed_wordtopic]
        word_term = (num / denom)

        ## combine and return 
        return wordtopic_term * word_term 

    def propose_new_par_wordtopic(self, doc_id, par_id, word_id):
        par_wordtopic_prob_vec = np.zeros(self.num_topics)
        for k in self.topics:
            par_wordtopic_prob_vec[k] = self.par_wordtopic_prob(k, doc_id, par_id, word_id)
        wordtopic = categorical(self.num_topics, p=par_wordtopic_prob_vec / par_wordtopic_prob_vec.sum())
        return wordtopic

    def block_sample_word_topic_and_switch(self):
        for doc_id, doc in tqdm(enumerate(self.docs), total=len(self.docs)):
            for par_id, par in enumerate(doc):
                old_partype = self.old_pardoc_to_type[(doc_id, par_id)]
                partype = self.pardoc_to_type[(doc_id, par_id)]
                ##
                for word_id, word in enumerate(par):
                    old_switching_var = self.switching_variable[(doc_id, par_id, word_id)]
                    wordtopic = self.word_to_topic[(doc_id, par_id, word_id)]
                    if old_switching_var == 0:
                        self.partype_by_wordtopic__wordtopic_counts[old_partype, wordtopic] -= 1
                        self.partype__wordtopic_counts[old_partype] -= 1
                    else:
                        self.doc_by_wordtopic__wordtopic_counts[doc_id, wordtopic] -= 1
                        self.doc__wordtopic_counts[doc_id] -= 1

                    s = self.sample_switching_variable(doc_id, par_id, word_id)
                    if s == 0:
                        wordtopic = self.propose_new_par_wordtopic(doc_id, par_id, word_id)
                        self.partype_by_wordtopic__wordtopic_counts[partype, wordtopic] += 1
                        self.partype__wordtopic_counts[partype] += 1
                    else:
                        wordtopic = self.propose_new_doc_wordtopic(doc_id, par_id, word_id)
                        self.doc_by_wordtopic__wordtopic_counts[doc_id, wordtopic] += 1
                        self.doc__wordtopic_counts[doc_id] += 1

                    self.switching_variable[(doc_id, par_id, word_id)] = s
                    self.word_to_topic[(doc_id, par_id, word_id)] = wordtopic


    def sample_pass(self):
        print('sampling doc-type...')
        self.sample_partype()
        print('sampling switch and word-topic...')
        self.block_sample_word_topic_and_switch()


    def joint_probability(self):
        pass


if __name__=="__main__":
    import argparse; p=argparse.ArgumentParser()
    # model params
    p.add_argument('-i', type=str, help="input directory.")
    p.add_argument('-o', type=str, help="output directory.")
    p.add_argument('-k', type=int, help="num topics.")
    p.add_argument('-p', type=int, help="num personas.")
    p.add_argument('-t', type=int, help="num iterations.")
    p.add_argument('--use-cached', action='store_true', dest='use_cached', help='use intermediate cached file.')
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

    sampler = BOW_Paragraph_GibbsSampler(docs=docs[:1000], vocab=vocab)

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
        if i % 10 == 0:
            pickle.dump(sampler, open(os.path.join(output_dir, 'trained-sampled-iter-%d.pkl' % (i+prev_iter)), 'wb'))
        sampler.sample_pass()

    ## done
    pickle.dump(sampler, open(os.path.join(output_dir, 'trained-sampled-iter-%d.pkl' % i), 'wb'))
