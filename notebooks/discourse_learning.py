import re
import glob
import os
import json
import pprint
import random
import nltk
import itertools
import logging
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from gensim.sklearn_api import d2vmodel, w2vmodel, ldamodel, text2bow
from gensim.models import doc2vec
from nltk import word_tokenize
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
import gensim
from nltk.corpus import verbnet as vn
from nltk.wsd import lesk
from collections import OrderedDict
from stanfordcorenlp import StanfordCoreNLP
import spacy
from hmmlearn import hmm

def get_data(path):
    source_files = glob.glob(path + "/*/*.sgm")
    data = {}
    for source in source_files:
        docID = os.path.splitext(os.path.basename(source))[0]
        with open(source) as f:
            data[docID] = clean_text("".join(f.readlines()))
    return data

def create_headline_dict(path):
    source_files = glob.glob(path + "/*.sgm")
    data = {}
    for source in source_files:
        docID = os.path.splitext(os.path.basename(source))[0]
        with open(source) as f:
            text = "".join(f.readlines())
            if text.find(r"<HEADLINE>") == -1:
                start = text.find("HEADLINE")
                end = text.find("\n", start)
                data[docID] = text[start + len("HEADLINE:"):end].strip()
            else:
                start = text.find(r"<HEADLINE>")
                end = text.find(r"</HEADLINE>", start)
                data[docID] = text[start + len(r"<HEADLINE>"):end].strip()
    return data

def clean_text(text):
    return re.sub("<.*?>", "", text)

def trim_verbnet_class(vnclass):
    return "-".join(vnclass.split("-")[:2])

def construct_verbnet_vector():
    vect = [trim_verbnet_class(k) for k in vn._class_to_fileid.keys()]
    return list(OrderedDict.fromkeys(vect))

def extract_verbnet_verbs(text):
    verbs = []
    toks = nltk.word_tokenize(text)
    for word, pos in nltk.pos_tag(toks):
        if 'V' in pos:
            synset = lesk(toks, word, 'v')
            if synset is not None:
                wnid = synset.lemmas()[0].key().strip(":")
                verbs.extend([trim_verbnet_class(k) for k in vn.classids(wordnetid=wnid)])
    return verbs

def extract_verbnet_feature(text, vector):
    verbs = extract_verbnet_verbs(text)
    vect = [0]*len(vector)
    for verb in verbs:
        if verb in vector:
            vect[vector.index(verb)] = 1
    return vect

def extract_verb_ratio_feature(text):
    verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    noun_count = 0
    pronoun_count = 0
    vect = [0]*len(verbs)
    toks = nltk.word_tokenize(text)
    for word, pos in nltk.pos_tag(toks):
        if pos in verbs:
            vect[verbs.index(pos)] += 1

    vect = [x/float(max(1, sum(vect))) for x in vect] # normalize with token length
    return vect

def extract_pos_ratio_feature(text):
    nouns = ['NN', 'NNP', 'NNPS', 'NNS']
    pronouns = ['PRP', 'PRP$']
    verbs = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    adverbs = ['RB', 'RBR', 'RBS']
    adjectives = ['JJ', 'JJR', 'JJS']
    wh_words = ['WDT', 'WP', 'WP$', 'WRB']
    cardinal = ['CD']
    
    noun_count = 0
    pronoun_count = 0
    vect = [0]*7
    toks = nltk.word_tokenize(text)
    for word, pos in nltk.pos_tag(toks):
        if pos in nouns:
            vect[0] += 1
        if pos in pronouns:
            vect[1] += 1
        if pos in verbs:
            vect[2] += 1
        if pos in adverbs:
            vect[3] += 1
        if pos in adjectives:
            vect[4] += 1
        if pos in wh_words:
            vect[5] += 1
        if pos in cardinal:
            vect[6] += 1

    vect = [x/float(len(toks)) for x in vect] # normalize with token length
    return vect

def extract_noun_ratio_feature(vect):
    return [(vect[0])/float(vect[0] + vect[1])]

#### COREF STUFF

def get_coref_chains(resp):
    return resp['corefs']

def sort_coref_chains(coref_chains, filterAnimate=False):
    chains = []
    for coref_id in coref_chains:
        chain = coref_chains[coref_id]
        if filterAnimate:
            if coref_vote_animacy(chain):
                chains.append((len(chain), chain))
            else:
                pass
        else:
            chains.append((len(chain), chain))
    chains.sort(key=lambda x: x[0], reverse=True)
    return [pair[1] for pair in chains] # return only the chains after sorting

def get_token_idx_map(resp):
    sents = resp['sentences']
    sent_dict = {}
    for sent in sents:
        sent_id = sent['index']
        sent_tokens = sent['tokens']
        token_vect = [''] * len(sent_tokens)
        for token in sent_tokens:
            token_vect[token['index'] - 1] = (token['characterOffsetBegin'], token['characterOffsetEnd'])
        sent_dict[sent_id] = token_vect
    return sent_dict

def get_ref_offsets(ref, token_map):
    sent_num = ref['sentNum'] - 1
    start = token_map[sent_num][ref['startIndex'] - 1][0]
    end = token_map[sent_num][ref['endIndex'] - 2][1]
    return (start, end)

def coref_vote_animacy(coref_chain):
    # list of references
    animacy = 0
    for ref in coref_chain:
        if ref['animacy'] == 'ANIMATE':
            animacy += 1
            
    if animacy/float(len(coref_chain)) > 0.5:
        # majority vote
        return True
    else:
        return False

def extract_doc_coref_feature(resp, filterAnimate, start, end):
    vect = [0]*5
    idx_map = get_token_idx_map(resp)
    coref_chains = sort_coref_chains(get_coref_chains(resp), filterAnimate=filterAnimate)[:5] # only check top 5
    for chain_idx in range(len(coref_chains)):
        chain = coref_chains[chain_idx]
        for ref in chain:
            ref_start, ref_end = get_ref_offsets(ref, idx_map)
            if end > ref_start > start and end > ref_end > start:
                vect[chain_idx] = 1
    return vect

def extract_paragraph_coref_feature(resp, filterAnimate):
    vect = [0]*5
    num_sents = len(resp['sentences'])
    coref_chains = sort_coref_chains(get_coref_chains(resp), filterAnimate=filterAnimate)[:5]
    for idx in range(len(coref_chains)):
        chain_length = len(coref_chains[idx])
        vect[idx] = chain_length/float(num_sents)
    return vect

def get_train_files(foldnum):
    return get_data("data/train/" + str(foldnum))

def get_test_files(foldnum):
    return get_data("data/test/" + str(foldnum))

def get_all_files():
    return {**get_data("data/train/" + str(0)), **get_data("data/test/" + str(0))}

def parse_annotations(docID, text, anno_path, nlp):
    vn_vector = construct_verbnet_vector()
    with open(anno_path) as f:
        annos = json.load(f)
    textPairs = []
    anno = annos[docID]
    pos = 1
    text_start = anno[0]["start"]
    text_end = anno[-1]["end"]
    #doc_resp = json.loads(nlp.annotate(text[text_start:text_end], properties={'annotators': 'tokenize,dcoref', 'pipelineLanguage':'en', 'outputFormat':'json'}))
    for p in anno:
        dtype = canonical_label_mapping(p["type"])
        paragraph = text[p["start"]:p["end"]].strip()

        #par_resp = json.loads(nlp.annotate(paragraph, properties={'annotators': 'dcoref', 'pipelineLanguage':'en', 'outputFormat':'json'}))

        feats = calculate_pos_features(p["start"], p["end"], len(text), len(anno), pos)
        feats['verbnet'] = extract_verbnet_feature(paragraph, vn_vector)
        feats['quote'] = extract_dm_quote_feature(paragraph)
        feats['coref_doc'] = [0]*5 #extract_doc_coref_feature(doc_resp, False, p["start"] - text_start, p["end"] - text_start) # offset
        feats['coref_doc_anim'] = [0]*5 #extract_doc_coref_feature(doc_resp, True, p["start"] - text_start, p["end"] - text_start) # offset
        feats['coref_para'] = [0]*5 #extract_paragraph_coref_feature(par_resp, False)
        feats['coref_para_anim'] = [0]*5 #extract_paragraph_coref_feature(par_resp, True)
        feats['pos_ratio'] = extract_pos_ratio_feature(paragraph)
        feats['noun_ratio'] = extract_noun_ratio_feature(feats['pos_ratio'])
        
        textPairs.append((paragraph, dtype, feats))
        pos += 1
    return textPairs

def create_feature_dict_file():
    train = get_train_files(0)
    test = get_test_files(0)
    train_pairs = []
    test_pairs = []

    feat_dict = {'parsed': []}
    with open("feat_dict.json") as f:
        feat_dict = json.load(f)

    print(len(feat_dict['parsed']))

    for i in train:
        if i not in feat_dict['parsed']:
            nlp = StanfordCoreNLP("stanford-corenlp-full-2018-02-27", memory='8g', timeout=10000)
            try:
                train_pairs.extend(parse_annotations(i, train[i], "anno_offsets_new.json", nlp))
                feat_dict['parsed'].append(i)
            except Exception as e:
                print(e)
                print(i)
            nlp.close()

    for i in test:
        if i not in feat_dict['parsed']:
            nlp = StanfordCoreNLP("stanford-corenlp-full-2018-02-27", memory='8g', timeout=10000)
            try:
                test_pairs.extend(parse_annotations(i, test[i], "anno_offsets_new.json", nlp))
                feat_dict['parsed'].append(i)
            except Exception as e:
                print(e)
                print(i)
            nlp.close()

    for i in train_pairs:
        feat_dict[i[0]] = i[2]

    for i in test_pairs:
        feat_dict[i[0]] = i[2]

    fname = "feat_dict" + ".json"
    with open(fname, 'w') as f:
        json.dump(feat_dict, f, indent=2)

    # this should construct the full feature dictionary (len 50)
    print(len(feat_dict))

def parse_annotations_no_featdict(docID, text, anno_path):
    with open(anno_path) as f:
        annos = json.load(f)
    textPairs = []
    anno = annos[docID]
    for p in anno:
        dtype = canonical_label_mapping(p["type"])
        paragraph = text[p["start"]:p["end"]].strip()
        
        textPairs.append((paragraph, dtype, p["start"], p["end"]))
    return textPairs

def parse_annotations_no_featdict_me(docID, text, anno_path):
    with open(anno_path) as f:
        annos = json.load(f)
    textPairs = []
    anno = annos[docID]
    for p in anno:
        dtype = canonical_label_mapping_me(p["type"])
        paragraph = text[p["start"]:p["end"]].strip()
        
        textPairs.append((paragraph, dtype, p["start"], p["end"]))
    return textPairs

def parse_annodict(docID, text, anno_path):
    with open(anno_path) as f:
        annos = json.load(f)
    annodict = {}
    anno = annos[docID]
    for p in anno:
        dtype = canonical_label_mapping(p["type"])
        paragraph = text[p["start"]:p["end"]].strip()
        if docID in annodict:
            annodict[docID].append((paragraph, dtype, p["start"], p["end"]))
        else:
            annodict[docID] = [(paragraph, dtype, p["start"], p["end"])]
    return annodict

def calculate_pos_features(start, end, textLength, annoLength, intPos):
    feats = {}
    feats['norm_size'] = [(end-start)/float(textLength)] # size %-wise
    feats['norm_char_pos'] = [end/float(textLength)] # rough ending position %-wise
    feats['norm_pos'] = [intPos/float(annoLength)] # rough integer position %-wise
    feats['verbnet'] = extract_verbnet_feature
    return feats

def extract_dm_quote_feature(text):
    # determines if there's a quotation mark
    if "\"" in text or "\'" in text:
        return [1]
    else:
        return [0]

def balance_data(textPairs):
    pairBalance = {0: [],
                   1: [],
                   2: [],
                   3: [],
                   4: [],
                   5: [],
                   6: [],
                   7: [],
                   8: []}

    for pair in textPairs:
        pairBalance[pair[1]].append(pair)

    minVal = float('inf')
    for k in pairBalance:
        if len(pairBalance[k]) < minVal:
            minVal = len(pairBalance[k])

    newPairs = []
    for k in pairBalance:
        newPairs.extend(pairBalance[k][minVal:])

    random.shuffle(newPairs)
    return newPairs

def canonical_label_mapping(dtype):
    dtype_map = {"LEAD": 0,
                 "MAIN": 1,
                 "CONS": 2,
                 "CIRC": 3,
                 "PREV": 4,
                 "HIST": 5,
                 "VERB": 6,
                 "EXPE": 7,
                 "EVAL": 8}
    return dtype_map[dtype]

def canonical_label_mapping_me(dtype):
    dtype_map = {"LEAD": 0,
                 "MAIN": 1,
                 "CONS": 0,
                 "CIRC": 0,
                 "PREV": 0,
                 "HIST": 0,
                 "VERB": 0,
                 "EXPE": 0,
                 "EVAL": 0}
    return dtype_map[dtype]

def inverse_label_mapping(dlabel):
    dlabel_map = {0: "LEAD",
                  1: "MAIN",
                  2: "CONS",
                  3: "CIRC",
                  4: "PREV",
                  5: "HIST",
                  6: "VERB",
                  7: "EXPE",
                  8: "EVAL"}
    return dlabel_map[dlabel]

def inverse_label_mapping_me(dlabel):
    dlabel_map = {0: "OTHER",
                  1: "MAIN"}
    return dlabel_map[dlabel]

def coarse_label_mapping(dlabel, level):
    dlabel = inverse_label_mapping(dlabel)
    dtype_map_1 = {"LEAD": "LEAD",
                   "MAIN": "EPISODE",
                   "CONS": "EPISODE",
                   "CIRC": "CONTEXT",
                   "PREV": "CONTEXT",
                   "HIST": "BACKGROUND",
                   "VERB": "COMMENTS",
                   "EXPE": "CONCLUSIONS",
                   "EVAL": "CONCLUSIONS"}
    dtype_map_2 = {"LEAD": "LEAD",
                   "MAIN": "EPISODE",
                   "CONS": "EPISODE",
                   "CIRC": "BACKGROUND",
                   "PREV": "BACKGROUND",
                   "HIST": "BACKGROUND",
                   "VERB": "COMMENTS",
                   "EXPE": "COMMENTS",
                   "EVAL": "COMMENTS"}
    dtype_map_3 = {"LEAD": "LEAD",
                   "MAIN": "SITUATION",
                   "CONS": "SITUATION",
                   "CIRC": "SITUATION",
                   "PREV": "SITUATION",
                   "HIST": "SITUATION",
                   "VERB": "COMMENTS",
                   "EXPE": "COMMENTS",
                   "EVAL": "COMMENTS"}
    if level == 1:
        return dtype_map_1[dlabel]
    elif level == 2:
        return dtype_map_2[dlabel]
    elif level == 3:
        return dtype_map_3[dlabel]
    else:
        return dlabel

def d2v_convert(texts):
    gendocs = []
    for i in range(len(texts)):
        gendocs.append(doc2vec.TaggedDocument(gensim.utils.simple_preprocess(texts[i]), [i]))
        #gendocs.append(doc2vec.TaggedDocument(word_tokenize(texts[i]), [i]))
    return gendocs

def cv5_test():
    sum_p = 0
    sum_r = 0
    sum_f1 = 0
    for fold in range(5):
        train = get_train_files(fold)
        test = get_test_files(fold)
        train_pairs = []
        test_pairs = []

        for i in train:
            train_pairs.extend(parse_annotations(i, train[i], "anno_offsets_new.json"))

        for i in test:
            test_pairs.extend(parse_annotations(i, test[i], "anno_offsets_new.json"))

        train_texts = [i[0] for i in train_pairs]
        train_labels = [i[1] for i in train_pairs]
        test_texts = [i[0] for i in test_pairs]
        test_labels = [i[1] for i in test_pairs]


        text_clf = Pipeline([('vect', CountVectorizer(stop_words="english")),
                             ('tfidf', TfidfTransformer()),
                             ('clf', svm.SVC(C=10, kernel="linear"))])

        text_clf = Pipeline([('d2v', d2vmodel.D2VTransformer(min_count=1)),
                             ('clf', svm.SVC(C=10, kernel="linear"))])

        d2v_train = d2v_convert(train_texts)
        d2v_test = d2v_convert(test_texts)

        text_clf.fit(d2v_train, train_labels)

        predicted = text_clf.predict(d2v_test)
        p = metrics.precision_score(test_labels, predicted, average="micro")
        r = metrics.recall_score(test_labels, predicted, average="micro")
        f1 = metrics.f1_score(test_labels, predicted, average="micro")

        sum_p += p
        sum_r += r
        sum_f1 += f1

        print("==========")
        print("Fold #" + str(fold + 1))
        print("Precision: " + str(p))
        print("Recall: " + str(r))
        print("F1: " + str(f1))

    print("==========")
    print("Total")
    print("Precision: " + str(sum_p/5.0))
    print("Recall: " + str(sum_r/5.0))
    print("F1: " + str(sum_f1/5.0))

class D2VTx(BaseEstimator, TransformerMixin):

    def __init__(self, dm=1, size=300, window=8, min_count=2, alpha=0.1, min_alpha=0.0001, steps=5):
        self.dm = dm
        self.size = size
        self.window = window
        self.min_count = min_count
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.steps = steps

    def fit(self, X, y=None):
        self.model = gensim.models.doc2vec.Doc2Vec(workers=8, vector_size=self.size, dm=self.dm, window=self.window, min_count=self.min_count)
        X_d2v = d2v_convert(X)
        self.model.build_vocab(X_d2v)
        self.model.train(X_d2v, total_examples=len(X_d2v), epochs=100)
        return self

    def transform(self, X):
        X_d2v = d2v_convert(X)
        res = np.zeros((len(X), self.size))
        for i, doc in enumerate(X_d2v):
            res[i] = self.model.infer_vector(doc.words, alpha=self.alpha, min_alpha=self.min_alpha, steps=self.steps)
        return res
        

class FeatExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, feat_dict, feats=None):
        self.feats = feats
        self.feat_dict = feat_dict # maps training text -> features

    def fit(self, X, y=None):
        return self # no fitting

    def transform(self, X):
        res = []
        for sample in X:
            f_list = []
            f_array = self.feat_dict[sample]
            if self.feats == None:
                for k in f_array:
                    f_list.extend(f_array[k])
            else:
                for f in self.feats:
                    if f in f_array:
                        f_list.extend(f_array[f])
                    else:
                        print("%s not in feature dict", f)
            res.append(f_list)
        return res

class FeatExtractor2(BaseEstimator, TransformerMixin):

    def __init__(self, feat_dict, feats=None):
        self.feats = feats
        self.feat_dict = feat_dict # maps training text -> features

    def fit(self, X, y=None):
        return self # no fitting

    def transform(self, X):
        res = []
        for sample in X:
            sample = "".join(sample.split("||")[1:])
            f_list = []
            f_array = self.feat_dict[sample]
            if self.feats == None:
                for k in f_array:
                    f_list.extend(f_array[k])
            else:
                for f in self.feats:
                    if f in f_array:
                        f_list.extend(f_array[f])
                    else:
                        print("%s not in feature dict", f)
            res.append(f_list)
        return res
        
def get_most_freq_type(train_labels):
    d = {}
    for label in train_labels:
        if label in d:
            d[label] += 1
        else:
            d[label] = 1

    mcount = 0
    mlabel = ""
    for label in d:
        if d[label] > mcount:
            mcount = d[label]
            mlabel = label

    return mlabel

def most_freq_classifier(mlabel, test_labels):
    return [mlabel]*len(test_labels)

def create_featdict():
    nlp = spacy.load('en')
    feat_dict = {}
    with open("new_feat_dict.json") as f:
        feat_dict = json.load(f)

    train = get_train_files(0)
    test = get_test_files(0)
    all_files = {**train, **test}

    headline_dict = create_headline_dict("flat")

    pairs = []
    for docID in all_files:
        text = all_files[docID]
        anno_path = "anno_offsets_new.json"
        with open(anno_path) as f:
            annos = json.load(f)
        textPairs = []
        anno = annos[docID]

        lead = ""

        for p in anno:
            if p["type"] == "LEAD":
                lead = text[p["start"]:p["end"]].strip()
                break
            

        doc1 = nlp(text)
        hl_doc = nlp(headline_dict[docID])
        lead_doc = nlp(lead)
        prev_label = -1
        p_list = []
        for p in anno:
            paragraph = text[p["start"]:p["end"]].strip()
            p_list.append(paragraph)
            doc2 = nlp(paragraph)

            feat_dict[paragraph]['sim_to_doc'] = [doc1.similarity(doc2)]
            feat_dict[paragraph]['verb_ratio'] = extract_verb_ratio_feature(paragraph)
            feat_dict[paragraph]['prev_label'] = [prev_label]
            feat_dict[paragraph]['hl_sim'] = [doc2.similarity(hl_doc)]
            feat_dict[paragraph]['lead_sim'] = [doc2.similarity(lead_doc)]
            prev_label = canonical_label_mapping(p["type"])

        for i in range(len(p_list)):
            max_sim = float("-inf")
            min_sim = float("inf")
            avg_sim = 0
            par = p_list[i]
            doc1 = nlp(par)
            for j in range(len(p_list)):
                if i != j:
                    doc2 = nlp(p_list[j])
                    sim = doc1.similarity(doc2)
                    if sim > max_sim:
                        max_sim = sim
                    if sim < min_sim:
                        min_sim = sim
                    avg_sim += sim
            feat_dict[par]['max_par_sim'] = [max_sim]
            feat_dict[par]['min_par_sim'] = [min_sim]
            feat_dict[par]['avg_par_sim'] = [avg_sim/float(len(p_list) - 1)]

        
        
    with open("new_feat_dict_x2.json", 'w') as f:
        json.dump(feat_dict, f, indent=2)
        
def parse_annotations(docID, text, anno_path, nlp):
    vn_vector = construct_verbnet_vector()
    with open(anno_path) as f:
        annos = json.load(f)
    textPairs = []
    anno = annos[docID]
    pos = 1
    text_start = anno[0]["start"]
    text_end = anno[-1]["end"]
    #doc_resp = json.loads(nlp.annotate(text[text_start:text_end], properties={'annotators': 'tokenize,dcoref', 'pipelineLanguage':'en', 'outputFormat':'json'}))
    for p in anno:
        dtype = canonical_label_mapping(p["type"])
        paragraph = text[p["start"]:p["end"]].strip()

        #par_resp = json.loads(nlp.annotate(paragraph, properties={'annotators': 'dcoref', 'pipelineLanguage':'en', 'outputFormat':'json'}))

        feats = calculate_pos_features(p["start"], p["end"], len(text), len(anno), pos)
        feats['verbnet'] = extract_verbnet_feature(paragraph, vn_vector)
        feats['quote'] = extract_dm_quote_feature(paragraph)
        feats['coref_doc'] = [0]*5 #extract_doc_coref_feature(doc_resp, False, p["start"] - text_start, p["end"] - text_start) # offset
        feats['coref_doc_anim'] = [0]*5 #extract_doc_coref_feature(doc_resp, True, p["start"] - text_start, p["end"] - text_start) # offset
        feats['coref_para'] = [0]*5 #extract_paragraph_coref_feature(par_resp, False)
        feats['coref_para_anim'] = [0]*5 #extract_paragraph_coref_feature(par_resp, True)
        feats['pos_ratio'] = extract_pos_ratio_feature(paragraph)
        feats['noun_ratio'] = extract_noun_ratio_feature(feats['pos_ratio'])
        
        textPairs.append((paragraph, dtype, feats))
        pos += 1
    return textPairs

def trans_laplace(train_dict, gold_dict):
    emission_dict = OrderedDict()
    for docID in train_dict:
        seq = []
        gold_seq = [-1]
        for i in train_dict[docID]:
            seq.append(i)
        for i in gold_dict[docID]:
            gold_seq.append(i[1])
        for i in range(len(seq)):
            prev_label = gold_seq[i-1]
            label = seq[i]
            if prev_label in emission_dict:
                if label in emission_dict[prev_label]:
                    emission_dict[prev_label][label] = emission_dict[prev_label][label] + 1
                else:
                    emission_dict[prev_label][label] = 1
            else:
                emission_dict[prev_label] = OrderedDict()
                emission_dict[prev_label][label] = 1

    emission_dict = OrderedDict(sorted(emission_dict.items(), key=lambda x: x[0]))
    
    for prev_label in emission_dict:
        for i in range(9):
            if i not in emission_dict[prev_label]:
                emission_dict[prev_label][i] = 0
        emission_dict[prev_label] = OrderedDict(sorted(emission_dict[prev_label].items(), key=lambda x: x[0]))
        row_sum = 0
        for label in emission_dict[prev_label]:
            row_sum += emission_dict[prev_label][label]
        for label in emission_dict[prev_label]:
            emission_dict[prev_label][label] = emission_dict[prev_label][label]/float(row_sum)

    emission_mat = []
    for prev_label in emission_dict:
        seq = []
        for label in emission_dict[prev_label]:
            seq.append(emission_dict[prev_label][label])
        emission_mat.append(seq)
    return emission_mat

def laplace(train_dict, gold_dict):
    emission_dict = OrderedDict()
    for docID in train_dict:
        seq = []
        gold_seq = []
        for i in train_dict[docID]:
            seq.append(i)
        for i in gold_dict[docID]:
            gold_seq.append(i[1])
        for i in range(len(seq)):
            gold_label = gold_seq[i]
            label = seq[i]
            if gold_label in emission_dict:
                if label in emission_dict[gold_label]:
                    emission_dict[gold_label][label] = emission_dict[gold_label][label] + 1
                else:
                    emission_dict[gold_label][label] = 1
            else:
                emission_dict[gold_label] = OrderedDict()
                emission_dict[gold_label][label] = 1

    emission_dict = OrderedDict(sorted(emission_dict.items(), key=lambda x: x[0]))
    
    for gold_label in emission_dict:
        for i in range(9):
            if i not in emission_dict[gold_label]:
                emission_dict[gold_label][i] = 0
        emission_dict[gold_label] = OrderedDict(sorted(emission_dict[gold_label].items(), key=lambda x: x[0]))
        row_sum = 0
        for label in emission_dict[gold_label]:
            row_sum += emission_dict[gold_label][label]
        for label in emission_dict[gold_label]:
            emission_dict[gold_label][label] = emission_dict[gold_label][label]/float(row_sum)

    emission_mat = []
    for gold_label in emission_dict:
        seq = []
        for label in emission_dict[gold_label]:
            seq.append(emission_dict[gold_label][label])
        emission_mat.append(seq)
    return emission_mat

def start_probs(train_dict):
    start_prob = OrderedDict()
    for docID in train_dict:
        seq = []
        label = train_dict[docID][0][1]
        if label in start_prob:
            start_prob[label] = start_prob[label] + 1
        else:
            start_prob[label] = 1

    for i in range(9):
        if i not in start_prob:
            start_prob[i] = 0
            
    start_prob = OrderedDict(sorted(start_prob.items(), key=lambda x: x[0]))

    for label in start_prob:
        start_prob[label] = start_prob[label]/float(len(train_dict))

    start_probs = []
    for label in start_prob:
        start_probs.append(start_prob[label])
    return start_probs
        

def hmm_test():
    sum_p = 0
    sum_r = 0
    sum_f1 = 0

    sum_p_M = 0
    sum_r_M = 0
    sum_f1_M = 0

    for fold in range(5):
        train = get_train_files(fold)
        test = get_test_files(fold)
        train_pairs = []
        test_pairs = []
        train_dict = {}
        test_dict = {}

        for i in train:
            train_pairs.extend(parse_annotations_no_featdict(i, train[i], "anno_offsets_new.json"))
            train_dict = {**train_dict, **parse_annodict(i, train[i], "anno_offsets_new.json")}

        for i in test:
            test_pairs.extend(parse_annotations_no_featdict(i, test[i], "anno_offsets_new.json"))
            test_dict = {**test_dict, **parse_annodict(i, test[i], "anno_offsets_new.json")}
        
        train_texts = [i[0] for i in train_pairs]
        train_labels = [i[1] for i in train_pairs]
        test_texts = [i[0] for i in test_pairs]
        test_labels = [i[1] for i in test_pairs]

        feat_dict = {}
        with open("feat_dict.json") as f:
            feat_dict = json.load(f)

        label_clf = CalibratedClassifierCV(svm.LinearSVC(C=10, class_weight='balanced'))

        label_pipe = Pipeline([
            ('feats', FeatExtractor(feat_dict, feats = ['prev_label'])),
            ('clf', label_clf)])

        label_pipe.fit(train_texts, train_labels)

        transmat = []

        for i in range(9):
            transmat.append(label_clf.predict_proba([[i]])[0])
        rtm = []
        for i in transmat:
            tmp = []
            for j in i:
                tmp.append(j)
            rtm.append(tmp)
        transmat = np.array(rtm)

        pp = pprint.PrettyPrinter(indent=2)

        startprob = start_probs(train_dict)

        model = hmm.MultinomialHMM(n_components=9)
        model.startprob_ = np.array(startprob)
        model.transmat_ = transmat

        text_clf = Pipeline([
            ('feats', FeatureUnion(transformer_list=[
                ('vect', CountVectorizer(stop_words="english")),
                ('tf_pipe', Pipeline([('vect', CountVectorizer(stop_words="english")),
                                      ('tfidf', TfidfTransformer())])),
                ('d2v', D2VTx(dm=0, min_alpha=0.01, min_count=5, steps=50)),
##                ('feats', FeatExtractor(feat_dict, feats = [
##                                                            'norm_size',
##                                                            'norm_char_pos',
##                                                            'norm_pos',
##                                                            'verbnet',
##                                                            'quote',
##                                                            'coref_doc',
##                                                            'coref_doc_anim',
##                                                            'coref_para',
##                                                            'coref_para_anim',
##                                                            'pos_ratio',
##                                                            'noun_ratio',
##                                                            'sim_to_doc',
##                                                            'verb_ratio',
##                                                            'prev_label'
##                                                            ])),
                ])),
            ('clf', svm.LinearSVC(C=10, class_weight='balanced'))])

        text_clf.fit(train_texts, train_labels)

        train_seqs = {}

        for docID in train_dict:
            train_seqs[docID] = []
            for pair in train_dict[docID]:
                train_seqs[docID].append(text_clf.predict([pair[0]])[0]) # predicted label

        em = laplace(train_seqs, train_dict)
        model.emissionprob_ = em
        model.n_features = 9
        

        predicted = text_clf.predict(test_texts)

        new_predicted = []
        new_gold = []
        for docID in test_dict:
            seq = []
            for i in test_dict[docID]:
                new_gold.append(i[1])
                seq.append(text_clf.predict([i[0]]))
            new_predicted.extend(list(model.decode(seq, algorithm='viterbi')[1]))            
            
##        predicted = new_predicted
##        test_labels = new_gold
        p = metrics.precision_score(test_labels, predicted, average="micro")
        r = metrics.recall_score(test_labels, predicted, average="micro")
        f1 = metrics.f1_score(test_labels, predicted, average="micro")

        sum_p += p
        sum_r += r
        sum_f1 += f1

        print("=== MICRO ===")
        print("Fold #" + str(fold + 1))
        print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % (p, r, f1))

        p = metrics.precision_score(test_labels, predicted, average="macro")
        r = metrics.recall_score(test_labels, predicted, average="macro")
        f1 = metrics.f1_score(test_labels, predicted, average="macro")

        sum_p_M += p
        sum_r_M += r
        sum_f1_M += f1

        print("=== MACRO ===")
        print("Fold #" + str(fold + 1))
        print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % (p, r, f1))

    print("*** TOTAL ***")

    print("=== MICRO ===")
    print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % ((sum_p/5.0), (sum_r/5.0), (sum_f1/5.0)))

    print("=== MACRO ===")
    print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % ((sum_p_M/5.0), (sum_r_M/5.0), (sum_f1_M/5.0)))


def hmm_test_2():
    sum_p = 0
    sum_r = 0
    sum_f1 = 0

    sum_p_M = 0
    sum_r_M = 0
    sum_f1_M = 0

    for fold in range(5):
        train = get_train_files(fold)
        test = get_test_files(fold)
        train_pairs = []
        test_pairs = []
        train_dict = {}
        test_dict = {}

        for i in train:
            train_pairs.extend(parse_annotations_no_featdict(i, train[i], "anno_offsets_new.json"))
            train_dict = {**train_dict, **parse_annodict(i, train[i], "anno_offsets_new.json")}

        for i in test:
            test_pairs.extend(parse_annotations_no_featdict(i, test[i], "anno_offsets_new.json"))
            test_dict = {**test_dict, **parse_annodict(i, test[i], "anno_offsets_new.json")}
        
        train_texts = [i[0] for i in train_pairs]
        train_labels = [i[1] for i in train_pairs]
        test_texts = [i[0] for i in test_pairs]
        test_labels = [i[1] for i in test_pairs]

        feat_dict = {}
        with open("feat_dict.json") as f:
            feat_dict = json.load(f)

        label_clf = CalibratedClassifierCV(svm.LinearSVC(C=10, class_weight='balanced'))

        label_pipe = Pipeline([
            ('feats', FeatExtractor(feat_dict, feats = ['prev_label'])),
            ('clf', label_clf)])

        label_pipe.fit(train_texts, train_labels)

        transmat = []

        for i in range(9):
            transmat.append(label_clf.predict_proba([[i]])[0])
        rtm = []
        for i in transmat:
            tmp = []
            for j in i:
                tmp.append(j)
            rtm.append(tmp)
        transmat = np.array(rtm)

        pp = pprint.PrettyPrinter(indent=2)

        startprob = start_probs(train_dict)

        model = hmm.MultinomialHMM(n_components=9)
        model.startprob_ = np.array(startprob)
        model.transmat_ = transmat

        text_clf = Pipeline([
            ('feats', FeatureUnion(transformer_list=[
                ('vect', CountVectorizer(stop_words="english")),
                ('tf_pipe', Pipeline([('vect', CountVectorizer(stop_words="english")),
                                      ('tfidf', TfidfTransformer())])),
                ('d2v', D2VTx(dm=0, min_alpha=0.01, min_count=5, steps=50)),
                ('feats', FeatExtractor(feat_dict, feats = [
##                                                            'norm_size',
##                                                            'norm_char_pos',
##                                                            'norm_pos',
##                                                            'verbnet',
##                                                            'quote',
##                                                            'coref_doc',
##                                                            'coref_doc_anim',
##                                                            'coref_para',
##                                                            'coref_para_anim',
##                                                            'pos_ratio',
##                                                            'noun_ratio',
##                                                            'sim_to_doc',
##                                                            'verb_ratio',
                                                            'prev_label'
                                                            ])),
                ])),
            ('clf', svm.LinearSVC(C=10, class_weight='balanced'))])

        text_clf.fit(train_texts, train_labels)

        train_seqs = {}

        docids = list(train_dict.keys())
        splits = np.array_split(docids, 5)

        for minifold in range(5):
            mclf = sklearn.base.clone(text_clf)
            trainids = []
            for i in range(5):
                if i != minifold:
                    trainids.extend(splits[i])
            testids = splits[minifold]
            mtrain_pairs = []
            mtest_pairs = []
            for did in trainids:
                mtrain_pairs.extend(train_dict[did])
            for did in testids:
                mtest_pairs.extend(train_dict[did])
            mtrain_texts = [i[0] for i in mtrain_pairs]
            mtrain_labels = [i[1] for i in mtrain_pairs]
            mtest_texts = [i[0] for i in mtest_pairs]
            mtest_labels = [i[1] for i in mtest_pairs]

            mclf.fit(mtrain_texts, mtrain_labels)

            for docID in testids:
                train_seqs[docID] = []
                for pair in train_dict[docID]:
                    train_seqs[docID].append(mclf.predict([pair[0]])[0])

        em = laplace(train_seqs, train_dict)
        model.emissionprob_ = em
        model.n_features = 9
        

        predicted = text_clf.predict(test_texts)

        new_predicted = []
        new_gold = []
        for docID in test_dict:
            seq = []
            for i in test_dict[docID]:
                new_gold.append(i[1])
                seq.append(text_clf.predict([i[0]]))
            new_predicted.extend(list(model.decode(seq, algorithm='viterbi')[1]))            
            
        predicted = new_predicted
        test_labels = new_gold
        p = metrics.precision_score(test_labels, predicted, average="micro")
        r = metrics.recall_score(test_labels, predicted, average="micro")
        f1 = metrics.f1_score(test_labels, predicted, average="micro")

        sum_p += p
        sum_r += r
        sum_f1 += f1

        print("=== MICRO ===")
        print("Fold #" + str(fold + 1))
        print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % (p, r, f1))

        p = metrics.precision_score(test_labels, predicted, average="macro")
        r = metrics.recall_score(test_labels, predicted, average="macro")
        f1 = metrics.f1_score(test_labels, predicted, average="macro")

        sum_p_M += p
        sum_r_M += r
        sum_f1_M += f1

        print("=== MACRO ===")
        print("Fold #" + str(fold + 1))
        print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % (p, r, f1))

    print("*** TOTAL ***")

    print("=== MICRO ===")
    print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % ((sum_p/5.0), (sum_r/5.0), (sum_f1/5.0)))

    print("=== MACRO ===")
    print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % ((sum_p_M/5.0), (sum_r_M/5.0), (sum_f1_M/5.0)))

def combine_feat_dicts(fd1, fd2):
    new_fd = {}
    for p in fd1:
        if p in fd2:
            new_fd[p] = {**fd1[p], **fd2[p]}
        else:
            new_fd[p] = fd1[p]
    with open("new_feat_dict.json", 'w') as f:
        json.dump(new_fd, f, indent=2)

def cv5_test_d2v():
    sum_p = 0
    sum_r = 0
    sum_f1 = 0

    sum_p_M = 0
    sum_r_M = 0
    sum_f1_M = 0

    sum_f1_c = {"LEAD": 0,
                "MAIN": 0,
                "CONS": 0,
                "CIRC": 0,
                "PREV": 0,
                "HIST": 0,
                "VERB": 0,
                "EXPE": 0,
                "EVAL": 0}

    label_counts = {"LEAD": 0,
                    "MAIN": 0,
                    "CONS": 0,
                    "CIRC": 0,
                    "PREV": 0,
                    "HIST": 0,
                    "VERB": 0,
                    "EXPE": 0,
                    "EVAL": 0}

    label_sum = 0

    labels_counted = False
    
    for fold in range(5):
        train = get_train_files(fold)
        test = get_test_files(fold)
        train_pairs = []
        test_pairs = []

        for i in train:
            train_pairs.extend(parse_annotations_no_featdict(i, train[i], "anno_offsets_new.json"))

        for i in test:
            test_pairs.extend(parse_annotations_no_featdict(i, test[i], "anno_offsets_new.json"))

        # train_pairs = balance_data(train_pairs) # balance training data; performs worse than balancing in SVC
        # random.shuffle(train_pairs) # shuffle training examples; minor on performance
        
        train_texts = [i[0] for i in train_pairs]
        train_labels = [i[1] for i in train_pairs]
        test_texts = [i[0] for i in test_pairs]
        test_labels = [i[1] for i in test_pairs]

        if not labels_counted:
            for i in train_labels + test_labels:
                label_counts[inverse_label_mapping(i)] += 1
                label_sum += 1

        feat_dict = {}
        with open("new_feat_dict.json") as f:
            feat_dict = json.load(f)

##        feat_dict_2 = {}
##        with open("rst_doc_features.json") as f:
##            feat_dict_2 = json.load(f)
##
##        combine_feat_dicts(feat_dict, feat_dict_2)
##        input("done")

        text_clf = Pipeline([
            ('feats', FeatureUnion(transformer_list=[
                ('vect', CountVectorizer(stop_words="english")),
                ('tf_pipe', Pipeline([('vect', CountVectorizer(stop_words="english")),
                                      ('tfidf', TfidfTransformer())])),
                ('d2v', D2VTx(dm=0, min_alpha=0.01, min_count=5, steps=50)),
                ('feats', FeatExtractor(feat_dict, feats = [
##                                                            'norm_size',
##                                                            'norm_char_pos',
##                                                            'norm_pos',
##                                                            'verbnet',
##                                                            'quote',
##                                                            'coref_doc',
##                                                            'coref_doc_anim',
##                                                            'coref_para',
##                                                            'coref_para_anim',
##                                                            'pos_ratio',
##                                                            'noun_ratio',
##                                                            'sim_to_doc',
##                                                            'verb_ratio',
                                                            'prev_label',
##                                                            'rstList',
##                                                            'rstCount',
                                                            'edu_short_dist',
                                                            'edu_long_dist',
                                                            'edu_common',
                                                            'edu_short_common',
                                                            'edu_long_common'
                                                            ])),
                ])),
            ('clf', svm.SVC(C=10, kernel="linear", class_weight='balanced'))])

        text_clf.fit(train_texts, train_labels)

        predicted = text_clf.predict(test_texts)

##        test_labels = list(map(lambda x: coarse_label_mapping(x, 3), test_labels))
##        predicted = list(map(lambda x: coarse_label_mapping(x, 3), predicted))
            
##        predicted = most_freq_classifier(get_most_freq_type(train_labels), test_labels)
        p = metrics.precision_score(test_labels, predicted, average="micro")
        r = metrics.recall_score(test_labels, predicted, average="micro")
        f1 = metrics.f1_score(test_labels, predicted, average="micro")

        sum_p += p
        sum_r += r
        sum_f1 += f1

        print("=== MICRO ===")
        print("Fold #" + str(fold + 1))
        print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % (p, r, f1))

        p = metrics.precision_score(test_labels, predicted, average="macro")
        r = metrics.recall_score(test_labels, predicted, average="macro")
        f1 = metrics.f1_score(test_labels, predicted, average="macro")

        sum_p_M += p
        sum_r_M += r
        sum_f1_M += f1

        print("=== MACRO ===")
        print("Fold #" + str(fold + 1))
        print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % (p, r, f1))

        class_res = metrics.f1_score(test_labels, predicted, average=None)
        for i in range(len(class_res)):
            sum_f1_c[inverse_label_mapping(i)] += class_res[i]

    print("*** TOTAL ***")

    print("=== MICRO ===")
    print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % ((sum_p/5.0), (sum_r/5.0), (sum_f1/5.0)))

    print("=== MACRO ===")
    print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % ((sum_p_M/5.0), (sum_r_M/5.0), (sum_f1_M/5.0)))

    print("=== CLASS ===")
    print({x: sum_f1_c[x]/5.0 for x in sum_f1_c})

    print("=== LABEL COUNTS ===")
    print(label_counts)
    print({x: label_counts[x]/float(label_sum) for x in label_counts})

def cv5_test_me():
    sum_p = 0
    sum_r = 0
    sum_f1 = 0

    sum_p_M = 0
    sum_r_M = 0
    sum_f1_M = 0

    sum_f1_c = {"OTHER": 0,
                "MAIN": 0}

    label_counts = {"OTHER": 0,
                    "MAIN": 0,}

    label_sum = 0

    labels_counted = False
    
    for fold in range(5):
        train = get_train_files(fold)
        test = get_test_files(fold)
        train_pairs = []
        test_pairs = []

        for i in train:
            train_pairs.extend(parse_annotations_no_featdict_me(i, train[i], "anno_offsets_new.json"))

        for i in test:
            test_pairs.extend(parse_annotations_no_featdict_me(i, test[i], "anno_offsets_new.json"))

        # train_pairs = balance_data(train_pairs) # balance training data; performs worse than balancing in SVC
        # random.shuffle(train_pairs) # shuffle training examples; minor on performance
        
        train_texts = [i[0] for i in train_pairs]
        train_labels = [i[1] for i in train_pairs]
        test_texts = [i[0] for i in test_pairs]
        test_labels = [i[1] for i in test_pairs]

        if not labels_counted:
            for i in train_labels + test_labels:
                label_counts[inverse_label_mapping_me(i)] += 1
                label_sum += 1

        feat_dict = {}
        with open("new_feat_dict_x2.json") as f:
            feat_dict = json.load(f)

##        feat_dict_2 = {}
##        with open("rst_doc_features.json") as f:
##            feat_dict_2 = json.load(f)
##
##        combine_feat_dicts(feat_dict, feat_dict_2)
##        input("done")

        text_clf = Pipeline([
            ('feats', FeatureUnion(transformer_list=[
##                ('vect', CountVectorizer(stop_words="english")),
                ('tf_pipe', Pipeline([('vect', CountVectorizer(stop_words="english")),
                                      ('tfidf', TfidfTransformer())])),
                ('d2v', D2VTx(dm=0, min_alpha=0.01, min_count=5, steps=50)),
                ('feats', FeatExtractor(feat_dict, feats = [
##                                                            'norm_size',
##                                                            'norm_char_pos',
                                                            'norm_pos',
##                                                            'verbnet',
##                                                            'quote',
##                                                            'coref_doc',
##                                                            'coref_doc_anim',
##                                                            'coref_para',
##                                                            'coref_para_anim',
##                                                            'pos_ratio',
##                                                            'noun_ratio',
                                                            'sim_to_doc',
                                                            'verb_ratio',
##                                                            'prev_label',
##                                                            'rstList',
##                                                            'rstCount',
##                                                            'edu_short_dist',
##                                                            'edu_long_dist',
##                                                            'edu_common',
                                                            'edu_short_common',
##                                                            'edu_long_common',
##                                                            'max_par_sim',
##                                                            'min_par_sim',
                                                            'avg_par_sim',
                                                            'hl_sim',
                                                            'lead_sim'
                                                            ])),
                ])),
            ('clf', svm.SVC(C=10, kernel="linear", class_weight='balanced'))])

        text_clf.fit(train_texts, train_labels)

        predicted = text_clf.predict(test_texts)

##        test_labels = list(map(lambda x: coarse_label_mapping(x, 3), test_labels))
##        predicted = list(map(lambda x: coarse_label_mapping(x, 3), predicted))
            
##        predicted = most_freq_classifier(get_most_freq_type(train_labels), test_labels)
        p = metrics.precision_score(test_labels, predicted, average="micro")
        r = metrics.recall_score(test_labels, predicted, average="micro")
        f1 = metrics.f1_score(test_labels, predicted, average="micro")

        sum_p += p
        sum_r += r
        sum_f1 += f1

        print("=== MICRO ===")
        print("Fold #" + str(fold + 1))
        print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % (p, r, f1))

        p = metrics.precision_score(test_labels, predicted, average="macro")
        r = metrics.recall_score(test_labels, predicted, average="macro")
        f1 = metrics.f1_score(test_labels, predicted, average="macro")

        sum_p_M += p
        sum_r_M += r
        sum_f1_M += f1

        print("=== MACRO ===")
        print("Fold #" + str(fold + 1))
        print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % (p, r, f1))

        class_res = metrics.f1_score(test_labels, predicted, average=None)
        for i in range(len(class_res)):
            sum_f1_c[inverse_label_mapping_me(i)] += class_res[i]

    print("*** TOTAL ***")

    print("=== MICRO ===")
    print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % ((sum_p/5.0), (sum_r/5.0), (sum_f1/5.0)))

    print("=== MACRO ===")
    print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % ((sum_p_M/5.0), (sum_r_M/5.0), (sum_f1_M/5.0)))

    print("=== CLASS ===")
    print({x: sum_f1_c[x]/5.0 for x in sum_f1_c})

    print("=== LABEL COUNTS ===")
    print(label_counts)
    print({x: label_counts[x]/float(label_sum) for x in label_counts})

def tree_test():
    sum_p = 0
    sum_r = 0
    sum_f1 = 0

    sum_p_M = 0
    sum_r_M = 0
    sum_f1_M = 0

    sum_f1_c = {"LEAD": 0,
                "MAIN": 0,
                "CONS": 0,
                "CIRC": 0,
                "PREV": 0,
                "HIST": 0,
                "VERB": 0,
                "EXPE": 0,
                "EVAL": 0}

    label_counts = {"LEAD": 0,
                    "MAIN": 0,
                    "CONS": 0,
                    "CIRC": 0,
                    "PREV": 0,
                    "HIST": 0,
                    "VERB": 0,
                    "EXPE": 0,
                    "EVAL": 0}

    label_sum = 0

    labels_counted = False
    
    for fold in range(5):
        train = get_train_files(fold)
        test = get_test_files(fold)
        train_pairs = []
        test_pairs = []

        for i in train:
            train_pairs.extend(parse_annotations_no_featdict(i, train[i], "anno_offsets_new.json"))

        for i in test:
            test_pairs.extend(parse_annotations_no_featdict(i, test[i], "anno_offsets_new.json"))

        # train_pairs = balance_data(train_pairs) # balance training data; performs worse than balancing in SVC
        # random.shuffle(train_pairs) # shuffle training examples; minor on performance
        
        train_texts = [i[0] for i in train_pairs]
        train_labels = [i[1] for i in train_pairs]
        test_texts = [i[0] for i in test_pairs]
        test_labels = [i[1] for i in test_pairs]

        if not labels_counted:
            for i in train_labels + test_labels:
                label_counts[inverse_label_mapping(i)] += 1
                label_sum += 1

        feat_dict = {}
        with open("feat_dict.json") as f:
            feat_dict = json.load(f)

        text_clf = Pipeline([
            ('feats', FeatureUnion(transformer_list=[
                ('vect', CountVectorizer(stop_words="english")),
                ('tf_pipe', Pipeline([('vect', CountVectorizer(stop_words="english")),
                                      ('tfidf', TfidfTransformer())])),
                ('d2v', D2VTx(dm=0, min_alpha=0.01, min_count=5, steps=50)),
                ('feats', FeatExtractor(feat_dict, feats = [
##                                                            'norm_size',
##                                                            'norm_char_pos',
##                                                            'norm_pos',
##                                                            'verbnet',
##                                                            'quote',
##                                                            'coref_doc',
##                                                            'coref_doc_anim',
##                                                            'coref_para',
##                                                            'coref_para_anim',
##                                                            'pos_ratio',
##                                                            'noun_ratio',
##                                                            'sim_to_doc',
##                                                            'verb_ratio',
                                                            'prev_label',
                                                            ])),
                ])),
##            ('clf', tree.DecisionTreeClassifier(class_weight='balanced'))
            ('clf', ensemble.RandomForestClassifier(class_weight='balanced', n_estimators=50))
            ])

        text_clf.fit(train_texts, train_labels)

        predicted = text_clf.predict(test_texts)

##        test_labels = list(map(lambda x: coarse_label_mapping(x, 3), test_labels))
##        predicted = list(map(lambda x: coarse_label_mapping(x, 3), predicted))
            
##        predicted = most_freq_classifier(get_most_freq_type(train_labels), test_labels)
        p = metrics.precision_score(test_labels, predicted, average="micro")
        r = metrics.recall_score(test_labels, predicted, average="micro")
        f1 = metrics.f1_score(test_labels, predicted, average="micro")

        sum_p += p
        sum_r += r
        sum_f1 += f1

        print("=== MICRO ===")
        print("Fold #" + str(fold + 1))
        print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % (p, r, f1))

        p = metrics.precision_score(test_labels, predicted, average="macro")
        r = metrics.recall_score(test_labels, predicted, average="macro")
        f1 = metrics.f1_score(test_labels, predicted, average="macro")

        sum_p_M += p
        sum_r_M += r
        sum_f1_M += f1

        print("=== MACRO ===")
        print("Fold #" + str(fold + 1))
        print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % (p, r, f1))

        class_res = metrics.f1_score(test_labels, predicted, average=None)
        for i in range(len(class_res)):
            sum_f1_c[inverse_label_mapping(i)] += class_res[i]

    print("*** TOTAL ***")

    print("=== MICRO ===")
    print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % ((sum_p/5.0), (sum_r/5.0), (sum_f1/5.0)))

    print("=== MACRO ===")
    print("P, R, F1:\t%0.3f\t%0.3f\t%0.3f" % ((sum_p_M/5.0), (sum_r_M/5.0), (sum_f1_M/5.0)))

    print("=== CLASS ===")
    print({x: sum_f1_c[x]/5.0 for x in sum_f1_c})

    print(label_counts)

    print({x: label_counts[x]/float(label_sum) for x in label_counts})

def generate_random_labels(path):
    anno_json = {}
    labels = ["LABEL_" + str(i) for i in range(1)] #["LEAD", "MAIN", "CONS", "CIRC", "PREV", "HIST", "VERB", "EXPE", "EVAL"]
##    f1 = 0
##    f1_m = 0
    for fold in range(5):
        train = get_train_files(fold)
        test = get_test_files(fold)
        train_pairs = []
        test_pairs = []

        for i in train:
            train_pairs.extend(parse_annotations_no_featdict(i, train[i], "anno_offsets_new.json"))

        for i in test:
            test_pairs.extend(parse_annotations_no_featdict(i, test[i], "anno_offsets_new.json"))
        
        train_texts = [i[0] for i in train_pairs]
        train_labels = [i[1] for i in train_pairs]
        test_texts = [i[0] for i in test_pairs]
        test_labels = [i[1] for i in test_pairs]

        predicted_labels = []

        for docID in test:
            paragraph_annos = parse_annotations_no_featdict(docID, test[docID], "anno_offsets_new.json")
            docList = []
            for p_anno in paragraph_annos:
                p_json = {}
                p_json["start"] = p_anno[2]
                p_json["end"] = p_anno[3]
                choice = random.choice(labels)
                p_json["type"] = choice
##                predicted_labels.append(canonical_label_mapping(choice))
                docList.append(p_json)
            anno_json[docID] = docList

##        f1 += metrics.f1_score(test_labels, predicted_labels, average="micro")
##        f1_m += metrics.f1_score(test_labels, predicted_labels, average="macro")

##    print(f1/5.0)
##    print(f1_m/5.0)
    with open(path, 'w') as f:
        json.dump(anno_json, f, indent=2)

    print(", ".join(labels))


def predict_labels(path):
    anno_json = {}
    for fold in range(5):
        train = get_train_files(fold)
        test = get_test_files(fold)
        train_pairs = []
        test_pairs = []

        for i in train:
            train_pairs.extend(parse_annotations_no_featdict(i, train[i], "anno_offsets_new.json"))

        

        # train_pairs = balance_data(train_pairs) # balance training data; performs worse than balancing in SVC
        # random.shuffle(train_pairs) # shuffle training examples; minor on performance
        
        train_texts = [i[0] for i in train_pairs]
        train_labels = [i[1] for i in train_pairs]

        feat_dict = {}
        with open("feat_dict.json") as f:
            feat_dict = json.load(f)

        text_clf = Pipeline([
            ('feats', FeatureUnion(transformer_list=[
                ('vect', CountVectorizer(stop_words="english")),
##                ('tf_pipe', Pipeline([('vect', CountVectorizer(stop_words="english")),
##                                      ('tfidf', TfidfTransformer())])),
##                ('d2v', D2VTx(dm=0, min_alpha=0.01, min_count=5, steps=50)),
##                ('feats', FeatExtractor(feat_dict, feats = ['norm_size',
##                                                            'norm_char_pos',
##                                                            'norm_pos',
##                                                            'verbnet',
##                                                            'quote',
##                                                            'coref_doc',
##                                                            'coref_doc_anim',
##                                                            'coref_para',
##                                                            'coref_para_anim',
##                                                            'pos_ratio',
##                                                            'noun_ratio'
##                                                            ])),
                ])),
            ('clf', svm.SVC(C=10, kernel="linear", class_weight='balanced'))])

        text_clf.fit(train_texts, train_labels)

        for docID in test:
            paragraph_annos = parse_annotations_no_featdict(docID, test[docID], "anno_offsets_new.json")
            docList = []
            for p_anno in paragraph_annos:
                p_json = {}
                p_json["start"] = p_anno[2]
                p_json["end"] = p_anno[3]
                p_json["type"] = inverse_label_mapping(int(text_clf.predict([p_anno[0]])[0]))
                docList.append(p_json)
            anno_json[docID] = docList
            
    with open(path, 'w') as f:
        json.dump(anno_json, f, indent=2)

def generate_feature_combos():
    combos = [[]]
    x1 = ['norm_size', 'norm_pos']
    toggles = ['verbnet', 'quote', 'coref_doc', 'coref_doc_anim', 'coref_para', 'coref_para_anim', 'pos_ratio', 'noun_ratio']
    for i in range(0, len(toggles) + 1):
        for subset in itertools.combinations(toggles, i):
            combos.append(x1 + list(subset))
    print(combos[0])
    print(len(combos))
    return combos

def generate_better_combos():
    combos = []
    toggles = [['norm_size', 'norm_pos'], ['prev_label'], ['quote'],
               ['coref_doc', 'coref_para'],
               ['pos_ratio', 'noun_ratio', 'verb_ratio']]
    for i in range(0, len(toggles) + 1):
        for subset in itertools.combinations(toggles, i):
            combos.extend(list(subset))
    print(len(combos))
    return combos

def gs_foldx(fold):
    param_grid = [{
                   'clf__n_estimators': [30],
                   'clf__criterion': ['gini'],
                   'clf__max_features': [None],
                   'clf__min_samples_split': [3],
                   'clf__min_samples_leaf': [1],
                   'clf__bootstrap': [True],
                   'clf__class_weight': ['balanced'],
                   'feats__feats__feats': generate_better_combos()
                   }]
                  
    train = get_train_files(fold)
    test = get_test_files(fold)
    train_pairs = []
    test_pairs = []

    for i in train:
        train_pairs.extend(parse_annotations_no_featdict(i, train[i], "anno_offsets_new.json"))

    for i in test:
        test_pairs.extend(parse_annotations_no_featdict(i, test[i], "anno_offsets_new.json"))

    train_texts = [i[0] for i in train_pairs]
    train_labels = [i[1] for i in train_pairs]
    test_texts = [i[0] for i in test_pairs]
    test_labels = [i[1] for i in test_pairs]

    feat_dict = {}
    with open("feat_dict.json") as f:
        feat_dict = json.load(f)

        text_clf = Pipeline([
            ('feats', FeatureUnion(transformer_list=[
                ('vect', CountVectorizer(stop_words="english")),
                ('tf_pipe', Pipeline([('vect', CountVectorizer(stop_words="english")),
                                      ('tfidf', TfidfTransformer())])),
                ('d2v', D2VTx(dm=0, min_alpha=0.01, min_count=5, steps=50)),
                ('feats', FeatExtractor(feat_dict, feats = [
                                                            'norm_size',
                                                            'norm_char_pos',
                                                            'norm_pos',
##                                                            'verbnet',
                                                            'quote',
##                                                            'coref_doc',
##                                                            'coref_doc_anim',
##                                                            'coref_para',
##                                                            'coref_para_anim',
##                                                            'pos_ratio',
##                                                            'noun_ratio',
##                                                            'sim_to_doc',
##                                                            'verb_ratio',
                                                            'prev_label',
                                                            ])),
                ])),
            ('clf', ensemble.RandomForestClassifier(n_jobs=-1))])

    gs_clf = GridSearchCV(text_clf, param_grid, scoring="f1_micro", n_jobs=-1, verbose=1)

    gs_clf = gs_clf.fit(train_texts, train_labels)

    predicted = gs_clf.predict(test_texts)
    p = metrics.precision_score(test_labels, predicted, average="micro")
    r = metrics.recall_score(test_labels, predicted, average="micro")
    f1 = metrics.f1_score(test_labels, predicted, average="micro")

    print("==========")
    print("GS Fold:" + str(fold + 1))
    print("Precision: " + str(p))
    print("Recall: " + str(r))
    print("F1: " + str(f1))

    print(gs_clf.best_score_)
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(gs_clf.best_params_)
    
def gs_fold(fold):
    param_grid = [{#'feats__tf_pipe__vect__ngram_range': [(1, 1), (1, 2)],
                   'feats__d2v__dm': [0],#, 1],
##                   'feats__d2v__size': [50, 300],
##                   'feats__d2v__window': [3, 8],
                   'feats__d2v__min_count': [2],# 5],
                   'feats__d2v__alpha': [0.01],# 0.1],
##                   'feats__d2v__min_alpha': [0.001, 0.0001, 0.00001],
                   'feats__d2v__steps': [5, 10],
                   'clf__C': [1, 10],
                   'clf__kernel': ['linear']},
                  {#'feats__tf_pipe__vect__ngram_range': [(1, 1), (1, 2)],
                   'feats__d2v__dm': [0, 1],
##                   'feats__d2v__size': [50, 300],
##                   'feats__d2v__window': [3, 8],
                   'feats__d2v__min_count': [2, 5],
                   'feats__d2v__alpha': [0.01, 0.1],
##                   'feats__d2v__min_alpha': [0.001, 0.0001, 0.00001],
                   'feats__d2v__steps': [5, 10],
                   'clf__C': [1, 10],
                   'clf__gamma': ['auto', 0.001, 0.01],
                   'clf__kernel': ['sigmoid', 'rbf']}]


    param_grid = [{'feats__feats__feats': generate_feature_combos()}]
                  
    train = get_train_files(fold)
    test = get_test_files(fold)
    train_pairs = []
    test_pairs = []

    for i in train:
        train_pairs.extend(parse_annotations_no_featdict(i, train[i], "anno_offsets_new.json"))

    for i in test:
        test_pairs.extend(parse_annotations_no_featdict(i, test[i], "anno_offsets_new.json"))

    train_texts = [i[0] for i in train_pairs]
    train_labels = [i[1] for i in train_pairs]
    test_texts = [i[0] for i in test_pairs]
    test_labels = [i[1] for i in test_pairs]

    feat_dict = {}
    with open("feat_dict.json") as f:
        feat_dict = json.load(f)

    text_clf = Pipeline([
        ('feats', FeatureUnion(transformer_list=[
            ('vect', CountVectorizer(stop_words="english")),
            ('tf_pipe', Pipeline([('vect', CountVectorizer(stop_words="english")),
                                  ('tfidf', TfidfTransformer())])),
            ('d2v', D2VTx(dm=0, min_alpha=0.01, min_count=5, steps=50)),
            ('feats', FeatExtractor(feat_dict)),
            ])),
        ('clf', svm.SVC(C=10, kernel="linear", class_weight='balanced'))])

    gs_clf = GridSearchCV(text_clf, param_grid, scoring="f1_micro", n_jobs=-1, verbose=1)

    gs_clf = gs_clf.fit(train_texts, train_labels)

    predicted = gs_clf.predict(test_texts)
    p = metrics.precision_score(test_labels, predicted, average="micro")
    r = metrics.recall_score(test_labels, predicted, average="micro")
    f1 = metrics.f1_score(test_labels, predicted, average="micro")

    print("==========")
    print("GS Fold:" + str(fold + 1))
    print("Precision: " + str(p))
    print("Recall: " + str(r))
    print("F1: " + str(f1))

    print(gs_clf.best_score_)
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(gs_clf.best_params_)

def label_stats(anno_path):
    with open(anno_path) as f:
        annos = json.load(f)
    d = {}
    c = 0
    pc = []
    for docID in annos:
        pc.append(len(annos[docID]))
        for p in annos[docID]:
            c += 1
            if p["type"] in d:
                d[p["type"]] = d[p["type"]] + 1
            else:
                d[p["type"]] = 1

    print(d)
    print(c)
    print(np.mean(pc))
    print(np.std(pc))

def generate_section_text(anno_path, out_path):
    vn_vector = construct_verbnet_vector()
    with open(anno_path) as f:
        annos = json.load(f)

    data = get_all_files()
    fnames = []

    for docID in data:
        anno = annos[docID]
        text = data[docID]
        pcount = 0
        for p in anno:
            fname = docID + "-" + str(pcount) + ".txt"
            fnames.append(fname)
            with open(out_path + "/" + fname, 'w') as f:
                f.write(text[p["start"]:p["end"]].strip())
            pcount += 1

    with open(out_path + "/filenames.txt", 'w') as f:
        f.write("\n".join(fnames))
    print("done")


def update_rst_features(anno_path, rst_path, out_path):
    vn_vector = construct_verbnet_vector()
    with open(anno_path) as f:
        annos = json.load(f)

    with open(rst_path) as f:
        rst_feats = json.load(f)

    data = get_all_files()
    fnames = []

    new_feats = {}

    for docID in data:
        anno = annos[docID]
        text = data[docID]
        pcount = 0
        for p in anno:
            ptext = text[p["start"]:p["end"]].strip()
            new_feats[ptext] = rst_feats[docID][str(pcount)]
            pcount += 1

    with open(out_path + "/new_rst_features.json", 'w') as f:
        json.dump(new_feats, f, indent=2)

if __name__ == "__main__":
##    pass
##    update_rst_features("anno_offsets_new.json", "rst_features.json", ".")
##    label_stats("anno_offsets_new.json")
##    cv5_test_d2v()
    cv5_test_me()
##    tree_test()
##    hmm_test_2()
##    for i in range(5):
##        gs_foldx(i)
    #create_feature_dict_file()
##    predict_labels("anno_offsets_bow.json")
##    generate_random_labels("../joint_ere_release/anno_offsets_rand.json")
##    create_featdict()
