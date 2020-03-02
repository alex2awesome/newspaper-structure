import json, os
import sampler_cy

with open('data/doc_vecs.json') as f:
    doc_strs =  f.read().split('\n')
    docs = []
    for idx, doc_str in enumerate(doc_strs):
#         if idx == 200:
#             break
        if doc_str:
            doc = json.loads(doc_str)
            docs.append(doc['paragraphs'])

vocab_fp = os.path.join('data/', 'vocab.txt')
vocab = open(vocab_fp).read().split('\n')
s2 = sampler_cy.BOW_Paragraph_GibbsSampler(docs=docs[:10], vocab=vocab)
s2.initialize()
s2.sample_pass()

s2.pythonize_vars()
print(s2.par_doc_to_type)
print(s2.partype_by_wordtopic)
a =1