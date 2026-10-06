# newspaper-structure

An early (2020) project on the discourse structure of news articles: how do newspapers structure a story, and can we recover that structure automatically? Each paragraph or sentence plays a role such as Lead, Main Event, Consequences, Circumstances, Previous Events, History, Verbal Reactions, Expectations or Evaluations, and we ask whether these roles can be learned from a small amount of annotated data. The repository tries three approaches: a Cython Gibbs sampler for a joint paragraph-type/word-topic model that infers paragraph roles with few or no labels; a BiLSTM-CRF sequence tagger over BERT sentence embeddings, adapted from Xiangci Li's scientific discourse tagger; and scikit-learn baselines. This is groundwork for Alexander Spangher's later newsworthiness research: understanding how articles are built is a step toward understanding what editors choose to cover and foreground. No paper draft is included.

## Layout

- `models/topic_model/` -- `sampler_cy.pyx` (the `BOW_Paragraph_GibbsSampler`), `setup.py` to compile it, `sampler_runner.py` to train, `run_kfold_cross-val.py` for 5-fold evaluation. Compiled binaries for Linux and Windows are checked in.
- `models/bilstm/` -- `discourse_tagger_generator_bert.py` and `run_discourse_tagger_bert.sh` (TensorFlow 1 / Keras), evaluation and attention-visualization notebooks; `data/` has train/val/test splits.
- `notebooks/` -- dated January-March 2020: exploring the annotated corpora, baseline models, running and analyzing the topic model, collecting new NYT/AP articles, preparing model inputs. `discourse_learning.py` holds feature extractors and sklearn pipelines.
- `app/` -- Flask annotation and validation interface (`app.py`, port 5005) for labeling sources quoted in articles by type and affiliation; writes to local JSON or Google Cloud Datastore.

## How to run

- Topic model: `cd models/topic_model && python setup.py build_ext --inplace`, then `python sampler_runner.py -i <input_dir> -o <output_dir> -k <num_topics> -p <num_paragraph_types> -t <iterations>`. The input directory needs `doc_vecs.json` (one document per line, paragraphs as word-id lists) and `vocab.txt`.
- Tagger: `cd models/bilstm && bash run_discourse_tagger_bert.sh` after pointing `--repfile` at BERT weights. Input is one sentence per line with a tab-separated label; blank lines separate documents.
- Annotation app: `cd app && python app.py`.

## Data

Expects a gitignored `data/` directory with the annotated corpora used in the notebooks: Finlayson et al.'s news-discourse annotations over ACE-style `.sgm` files, news-structure labels over Gigaword NYT articles from Ruihong Huang's group, and a sample of NYT front-page articles. Only the small tagger splits in `models/bilstm/data/` are included.

## Status

Notebooks date from early 2020; the repository was last touched in November 2023.
