#!python
#cython: boundscheck=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: wraparound=False

import numpy as np
cimport numpy as np 
from cython.parallel import prange
from libc.math cimport log, exp 
from libc.stdlib cimport rand, RAND_MAX
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
import cython 
import os
import json

DTYPE = np.intc

cdef double random_double(double max_num) nogil:
    cdef double n = rand()
    cdef double d = RAND_MAX + 1
    return n / d * max_num

cdef int categorical(int k, double[:] p) nogil:
    cdef double sum_of_weight = 0;
    cdef int i
    for i in range(k):
        sum_of_weight += p[i];
    cdef double rnd = random_double(sum_of_weight);
    for i in range(k):
        if rnd < p[i]:
            return i;
        rnd -= p[i];

cdef int bernoilli(double p) nogil:
    cdef double r = random_double(1)
    if r < p:
        return 1
    else:
        return 0


cdef class BOW_Paragraph_GibbsSampler():
    cdef int ***docs 
    cdef int[:] num_pars_per_doc
    cdef int **num_words_per_par_per_doc
    cdef int **labels
    cdef bint *doc_has_labels

    cdef int n_docs, n_graphs, n_vocab, num_partypes, num_topics, max_graphs_per_doc
    cdef double H_T, H_P, gamma, alpha, beta
    
    cdef int[:] partype_counts, doc__wordtopic_counts, partype__wordtopic_counts, wordtopic__word_counts
    cdef int[:, :] doc_by_wordtopic__wordtopic_counts, partype_by_wordtopic__wordtopic_counts, vocab_by_wordtopic__word_counts
    cdef int** pardoc_to_type
    cdef int** old_pardoc_to_type
    cdef int*** word_to_topic
    cdef int*** switching_variable
    cdef int*** old_switching_variable

    cdef public np.ndarray doc_by_wordtopic, partype_by_wordtopic, vocab_by_wordtopic, switching_variable_counts
    cdef public list par_doc_to_type

    cdef double[:] par_prob_vec, par_wordtopic_prob_vec, doc_wordtopic_prob_vec


    def __cinit__(
        self, list docs, list vocab, int num_partypes=10, int num_topics=25,
        double H_P_prior=1, double H_T_prior=1, double gamma=.7, double alpha=1., double beta=1.
    ):
        # copy word-level and paragraph-level data
        self.docs                                = <int***>PyMem_Malloc(len(docs) * sizeof(int**))
        self.word_to_topic                       = <int***>PyMem_Malloc(len(docs) * sizeof(int**))
        self.switching_variable                  = <int***>PyMem_Malloc(len(docs) * sizeof(int**))
        self.old_switching_variable              = <int***>PyMem_Malloc(len(docs) * sizeof(int**))
        self.num_words_per_par_per_doc           =  <int**>PyMem_Malloc(len(docs) * sizeof(int*))
        self.labels                              =  <int**>PyMem_Malloc(len(docs) * sizeof(int*))
        self.pardoc_to_type                      =  <int**>PyMem_Malloc(len(docs) * sizeof(int*))
        self.old_pardoc_to_type                  =  <int**>PyMem_Malloc(len(docs) * sizeof(int*))
        self.doc_has_labels                      =  <bint*>PyMem_Malloc(len(docs) * sizeof(bint))
        cdef int i, j, k
        ## document-level for-loop
        for i in range(len(docs)):
            self.docs[i]                         = <int**> PyMem_Malloc(len(docs[i]['paragraphs']) * sizeof(int*))
            self.word_to_topic[i]                = <int**> PyMem_Malloc(len(docs[i]['paragraphs']) * sizeof(int*))
            self.switching_variable[i]           = <int**> PyMem_Malloc(len(docs[i]['paragraphs']) * sizeof(int*))
            self.old_switching_variable[i]       = <int**> PyMem_Malloc(len(docs[i]['paragraphs']) * sizeof(int*))
            self.num_words_per_par_per_doc[i]    = <int*>  PyMem_Malloc(len(docs[i]['paragraphs']) * sizeof(int))
            self.labels[i]                       = <int*>  PyMem_Malloc(len(docs[i]['paragraphs']) * sizeof(int))
            self.pardoc_to_type[i]               = <int*>  PyMem_Malloc(len(docs[i]['paragraphs']) * sizeof(int))
            self.old_pardoc_to_type[i]           = <int*>  PyMem_Malloc(len(docs[i]['paragraphs']) * sizeof(int))
            self.doc_has_labels[i]               = docs[i]['has_labels']
            ## paragraph-level
            for j in range(len(docs[i]['paragraphs'])):
                self.docs[i][j]                      = <int*> PyMem_Malloc(len(docs[i]['paragraphs'][j]) * sizeof(int))
                self.word_to_topic[i][j]             = <int*> PyMem_Malloc(len(docs[i]['paragraphs'][j]) * sizeof(int))
                self.switching_variable[i][j]        = <int*> PyMem_Malloc(len(docs[i]['paragraphs'][j]) * sizeof(int))
                self.old_switching_variable[i][j]    = <int*> PyMem_Malloc(len(docs[i]['paragraphs'][j]) * sizeof(int))
                self.num_words_per_par_per_doc[i][j] = len(docs[i]['paragraphs'][j])
                self.labels[i][j]                    = docs[i]['labels'][j]
                ## word-level
                for k in range(len(docs[i]['paragraphs'][j])):
                    self.docs[i][j][k] = docs[i]['paragraphs'][j][k]
            
        self.num_pars_per_doc = np.array(list(map(lambda x: len(x['paragraphs']), docs)), dtype=DTYPE)
        self.n_docs = len(docs)
        self.n_graphs = sum(list(map(lambda x: len(x['paragraphs']), docs)))
        self.max_graphs_per_doc = max(list(map(lambda x: len(x['paragraphs']), docs)))
        self.n_vocab = len(vocab)

        # hyperparameters
        self.num_partypes = num_partypes 
        self.num_topics = num_topics

        self.par_prob_vec = np.ones(self.num_partypes)
        self.par_wordtopic_prob_vec = np.ones(self.num_topics)
        self.doc_wordtopic_prob_vec = np.ones(self.num_topics)

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
        self.partype_counts = np.zeros(self.num_partypes, dtype=DTYPE)

        ## background word-topic probabilities
        #### I(doc d has type t, word w has topic k)
        #### count_k_d_t_w
        self.doc__wordtopic_counts = np.zeros(self.n_docs, dtype=DTYPE)
        self.doc_by_wordtopic__wordtopic_counts = np.zeros((self.n_docs, self.num_topics), dtype=DTYPE)

        ## paragraph word-topic probabilities
        #### I(source S_d in doc d has type s, word w has topic k)
        #### count_k_d_s_Sd_w
        self.partype__wordtopic_counts = np.zeros(self.num_partypes, dtype=DTYPE)
        self.partype_by_wordtopic__wordtopic_counts = np.zeros((self.num_partypes, self.num_topics), dtype=DTYPE)

        ## word probabilities
        self.vocab_by_wordtopic__word_counts = np.zeros((self.n_vocab, self.num_topics), dtype=DTYPE)
        self.wordtopic__word_counts = np.zeros(self.num_topics, dtype=DTYPE)

    def initialize(self):
        ## probability vectors
        cdef int doc_id, par_id, word
        cdef int** doc
        cdef int* par
        cdef int partype, s
        ## iterate through documents
        for doc_id in range(self.n_docs):
            for par_id in range(self.num_pars_per_doc[doc_id]):
                if not self.doc_has_labels[doc_id]:
                    partype = categorical(self.num_partypes, p=self.par_prob_vec)
                else:
                    partype = self.labels[doc_id][par_id]

                self.partype_counts[partype] += 1
                self.pardoc_to_type[doc_id][par_id] = partype
                self.old_pardoc_to_type[doc_id][par_id] = partype

                for word_id in range(self.num_words_per_par_per_doc[doc_id][par_id]):
                    word = self.docs[doc_id][par_id][word_id]
                    s = bernoilli(self.gamma)
                    self.switching_variable[doc_id][par_id][word_id] = s
                    self.old_switching_variable[doc_id][par_id][word_id] = s

                    if s == 0:
                        ## set paragraph word-topics counts
                        word_topic = categorical(self.num_topics, p=self.par_wordtopic_prob_vec)
                        self.partype_by_wordtopic__wordtopic_counts[partype, word_topic] += 1
                        self.partype__wordtopic_counts[partype] += 1
                        self.vocab_by_wordtopic__word_counts[word, word_topic] += 1
                        self.wordtopic__word_counts[word_topic] += 1

                    else:
                        ## set document word-topic counts
                        word_topic = categorical(self.num_topics, p=self.doc_wordtopic_prob_vec)
                        self.doc_by_wordtopic__wordtopic_counts[doc_id, word_topic] += 1
                        self.doc__wordtopic_counts[doc_id] += 1
                        self.vocab_by_wordtopic__word_counts[word, word_topic] += 1
                        self.wordtopic__word_counts[word_topic] += 1

                    ## cache
                    self.word_to_topic[doc_id][par_id][word_id] = word_topic


    ###
    # sample par type
    ###
    cdef double partype_prob(self, int proposed_partype, int doc_id, int par_id) nogil:
        ## partype factor
        cdef double partype_term = log(self.H_T + self.partype_counts[proposed_partype])

        ## paragraph word_topic factor
        cdef double paragraph_wordtopic_term = 0
        cdef double log_denom = log(self.num_topics * self.H_P + self.partype__wordtopic_counts[proposed_partype])
        cdef int word_id, wordtopic
        cdef double num
        for word_id in range(self.num_words_per_par_per_doc[doc_id][par_id]):
            if paragraph_wordtopic_term < -100: ## hack... is there any other way to deal with very low probabilities?
                break
            if self.switching_variable[doc_id][par_id][word_id] == 0:
                wordtopic = self.word_to_topic[doc_id][par_id][word_id]
                num = self.H_P + self.partype_by_wordtopic__wordtopic_counts[proposed_partype, wordtopic]
                paragraph_wordtopic_term += (log(num) - log_denom)

        ## combine and return
        cdef double log_prob = partype_term + paragraph_wordtopic_term
        return exp(log_prob)

    cdef propose_new_partype(self, int doc_id, int par_id):
        cdef int t
        for t in prange(self.num_partypes, nogil=True):
            self.par_prob_vec[t] = self.partype_prob(t, doc_id, par_id)
        cdef int new_partype = categorical(self.num_partypes, p=self.par_prob_vec)
        return new_partype

    cdef void sample_partype(self):
        ## for each doc, decrement counts and resample
        cdef int** doc
        cdef int* par
        cdef int doc_id, par_id
        cdef int old_partype, new_partype
        for doc_id in range(self.n_docs):
            for par_id in range(self.num_pars_per_doc[doc_id]):
                if not self.doc_has_labels[doc_id]:
                    old_partype = self.pardoc_to_type[doc_id][par_id]
                    ## decrement
                    self.partype_counts[old_partype] -= 1
                    new_partype = self.propose_new_partype(doc_id, par_id)
                    ## increment
                    self.partype_counts[new_partype] += 1
                    ## cache
                    self.pardoc_to_type[doc_id][par_id] = new_partype
                    self.old_pardoc_to_type[doc_id][par_id] = old_partype

    ##
    # Sample switching variable...
    ##
    cdef double switching_prob_0(self, int partype, int wordtopic) nogil:
        '''Gamma = 0 is paragraph word.'''
        cdef double wordtopic_term = (
            (self.partype_by_wordtopic__wordtopic_counts[partype, wordtopic] + self.H_P)
        /
            (self.partype__wordtopic_counts[partype] + self.num_topics * self.H_P)
        )
        return self.gamma * wordtopic_term

    cdef double switching_prob_1(self, int doc_id, int wordtopic) nogil:
        '''Gamma = 1 is document word.'''
        cdef double wordtopic_term = (
            (self.doc_by_wordtopic__wordtopic_counts[doc_id, wordtopic] + self.alpha)
        /
            (self.doc__wordtopic_counts[doc_id] + self.num_topics * self.alpha)
        )
        return (1 - self.gamma) * wordtopic_term

    cdef int sample_switching_variable(self, int doc_id, int par_id, int word_id) nogil:
        cdef int partype = self.pardoc_to_type[doc_id][par_id]
        cdef int wordtopic = self.word_to_topic[doc_id][par_id][word_id]
        cdef double p_s_0 = self.switching_prob_0(partype, wordtopic)
        cdef double p_s_1 = self.switching_prob_1(doc_id, wordtopic)
        cdef double p = p_s_0 / (p_s_0 + p_s_1)
        cdef int new_switching_var = bernoilli(p)
        return new_switching_var

    ###
    # Sample word topic
    ###
    cdef double doc_wordtopic_prob(self, int proposed_wordtopic, int doc_id, int par_id, int word_id) nogil:
        # old_wordtopic = self.word_to_backgroundtopic[(doc_id, par_id, word_id)]
        cdef int word = self.docs[doc_id][par_id][word_id]

        ## source word_topic factor
        cdef double wordtopic_term = self.alpha + self.doc_by_wordtopic__wordtopic_counts[doc_id, proposed_wordtopic]

        ## word factor
        cdef double denom = self.n_vocab * self.beta + self.wordtopic__word_counts[proposed_wordtopic]
        cdef double num = self.beta + self.vocab_by_wordtopic__word_counts[word, proposed_wordtopic]
        cdef double word_term = (num / denom)

        ## combine and return
        return wordtopic_term * word_term

    cdef int propose_new_doc_wordtopic(self, int doc_id, int par_id, int word_id):
        cdef int k
        for k in prange(self.num_topics, nogil=True):
            self.doc_wordtopic_prob_vec[k] = self.doc_wordtopic_prob(k, doc_id, par_id, word_id)
        cdef int wordtopic = categorical(self.num_topics, p=self.doc_wordtopic_prob_vec)
        return wordtopic

    cdef double par_wordtopic_prob(self, int proposed_wordtopic, int doc_id, int par_id, int word_id) nogil:
        cdef int word = self.docs[doc_id][par_id][word_id]
        cdef int partype = self.pardoc_to_type[doc_id][par_id]

        ## source word_topic factor
        cdef double wordtopic_term = self.H_P + self.partype_by_wordtopic__wordtopic_counts[partype, proposed_wordtopic]

        ## word factor
        cdef double denom = self.n_vocab * self.beta + self.wordtopic__word_counts[proposed_wordtopic]
        cdef double num = self.beta + self.vocab_by_wordtopic__word_counts[word, proposed_wordtopic]
        cdef double word_term = (num / denom)

        ## combine and return
        return wordtopic_term * word_term

    cdef int propose_new_par_wordtopic(self, int doc_id, int par_id, int word_id):
        cdef int k
        for k in prange(self.num_topics, nogil=True):
            self.par_wordtopic_prob_vec[k] = self.par_wordtopic_prob(k, doc_id, par_id, word_id)
        cdef int wordtopic = categorical(self.num_topics, p=self.par_wordtopic_prob_vec)
        return wordtopic

    cdef void block_sample_word_topic_and_switch(self):
        cdef int** doc
        cdef int* par
        cdef int doc_id, par_id, word
        cdef int old_partype, partype, old_switching_variable, wordtopic, s
        for doc_id in range(self.n_docs):
            for par_id in range(self.num_pars_per_doc[doc_id]):
                old_partype = self.old_pardoc_to_type[doc_id][par_id]
                partype = self.pardoc_to_type[doc_id][par_id]
                ##
                for word_id in range(self.num_words_per_par_per_doc[doc_id][par_id]):
                    old_switching_var = self.switching_variable[doc_id][par_id][word_id]
                    wordtopic = self.word_to_topic[doc_id][par_id][word_id]
                    if old_switching_var == 0:
                        self.partype_by_wordtopic__wordtopic_counts[old_partype, wordtopic] -= 1
                        self.partype__wordtopic_counts[old_partype] -= 1
                    else:
                        self.doc_by_wordtopic__wordtopic_counts[doc_id, wordtopic] -= 1
                        self.doc__wordtopic_counts[doc_id] -= 1

                    s = self.sample_switching_variable(doc_id, par_id, word_id)
                    self.switching_variable[doc_id][par_id][word_id] = s
                    if s == 0:
                        wordtopic = self.propose_new_par_wordtopic(doc_id, par_id, word_id)
                        self.partype_by_wordtopic__wordtopic_counts[partype, wordtopic] += 1
                        self.partype__wordtopic_counts[partype] += 1
                    else:
                        wordtopic = self.propose_new_doc_wordtopic(doc_id, par_id, word_id)
                        self.doc_by_wordtopic__wordtopic_counts[doc_id, wordtopic] += 1
                        self.doc__wordtopic_counts[doc_id] += 1

                    self.switching_variable[doc_id][par_id][word_id] = s
                    self.word_to_topic[doc_id][par_id][word_id] = wordtopic

    def pythonize_vars(self):
        ## convert pointer to pointers => list of lists
        self.par_doc_to_type = []
        for i in range(self.n_docs):
            self.par_doc_to_type.append([])
            for j in range(self.num_pars_per_doc[i]):
                self.par_doc_to_type[i].append(self.pardoc_to_type[i][j])

        ## memoryview => numpy array
        self.doc_by_wordtopic = np.asarray(self.doc_by_wordtopic__wordtopic_counts)
        self.partype_by_wordtopic = np.asarray(self.partype_by_wordtopic__wordtopic_counts)
        self.vocab_by_wordtopic = np.asarray(self.vocab_by_wordtopic__word_counts)
        self.switching_variable_counts = np.zeros((self.n_vocab, 2), dtype=DTYPE)
        for i in range(self.n_docs):
            for j in range(self.num_pars_per_doc[i]):
                for k in range(self.num_words_per_par_per_doc[i][j]):
                    word = self.docs[i][j][k]
                    s = self.switching_variable[i][j][k]
                    self.switching_variable_counts[word, s] += 1


    def save_state(self, output_dir):
        self.pythonize_vars()
        with open(os.path.join(output_dir, 'pardoc_to_type.txt'), 'w') as f:
            json.dump(self.par_doc_to_type, f)
        np.savetxt(os.path.join(output_dir, 'doc_by_wordtopic.txt'), self.doc_by_wordtopic)
        np.savetxt(os.path.join(output_dir, 'partype_by_wordtopic.txt'), self.partype_by_wordtopic)
        np.savetxt(os.path.join(output_dir, 'vocab_by_wordtopic.txt'), self.vocab_by_wordtopic)
        np.savetxt(os.path.join(output_dir, 'switching_variable_counts.txt'), self.switching_variable_counts)


    def __dealloc__(self):
        PyMem_Free(self.docs)
        PyMem_Free(self.num_words_per_par_per_doc)
        PyMem_Free(self.switching_variable)
        PyMem_Free(self.old_switching_variable)
        PyMem_Free(self.word_to_topic)
        PyMem_Free(self.labels)
        PyMem_Free(self.pardoc_to_type)
        PyMem_Free(self.old_pardoc_to_type)
        PyMem_Free(self.doc_has_labels)

    def sample_pass(self):
        self.sample_partype()
        self.block_sample_word_topic_and_switch()

    def joint_probability(self):
        pass