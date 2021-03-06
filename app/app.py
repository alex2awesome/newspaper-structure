from flask import Flask, render_template, request
from google.cloud import datastore
from collections import defaultdict
import numpy as np
import glob
import json, os

app = Flask(__name__)

def get_client():
    try:
        return datastore.Client()
    except:
        import os
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/alexa/google-cloud/usc-research-c087445cf499.json'
        return datastore.Client()


@app.route('/')
def hello():
    return "Hello World!"

def get_user(email, client):
    """Get user. Create new entity if doesn't exist."""
    user_key = client.key('user', email)
    user = client.get(user_key)
    if user:
        return user
    e = datastore.Entity(key=user_key)
    e.update({'total_tasks': 0})
    client.put(e)
    return e

@app.route('/get_user_stats', methods=['POST'])
def get_user_stats():
    request_data = request.get_json()
    client = get_client()
    user_email = request_data.get('user_email', '')
    user = get_user(user_email, client)
    return str(user['total_tasks'])

###
#   annotation
###
def get_data_json(pretagged=True, n_per_page=1):
    if pretagged == True:
        input_file = 'data/pretagged_records.json'
        output_dir = 'data/pretagged_output/'
    else:
        input_file = 'data/untagged_records.json'
        output_dir = 'data/untagged_output/'
    ## 
    with open(input_file) as f:
        results = json.load(f)
    num_tagged_batches = len(os.listdir(output_dir))
    results = results[num_tagged_batches * n_per_page : num_tagged_batches * n_per_page + n_per_page]
    return results

def get_data_gcloud():
    client = get_client()
    query = client.query(kind='law-annotation-unmarked')
    query.add_filter('done', '=', False)
    results = list(query.fetch(limit=10))
    return results

@app.route('/render_annotation_experiment')
def render_annotation():
    results = get_data_json(pretagged=False)
    num_laws = len(results)
    return render_template(
        'task-annotation-slim.html',
        paper_count=num_laws,
        input=results
    )


def post_data_gcloud(data):
    client = get_client()
    for key, val in output_dict.items():
        ## new
        marked_k = client.key('newsstructure-annotation-marked', key)
        marked_e = datastore.Entity(marked_k, exclude_from_indexes=['data',])
        marked_e.update({'data': val})
        client.put(marked_e)
        ## update old
        unmarked_k = client.key('newsstructure-annotation-unmarked', key)
        unmarked_e = client.get(unmarked_k)
        if unmarked_e:
            unmarked_e['done'] = True
            client.put(unmarked_e)
    return True    

def post_data_local(data, pretagged=True):
    if pretagged:
        output_dir = 'pretagged_output'
    else:
        output_dir = 'untagged_output'
    num_tagged_batches = len(os.listdir('data/' + output_dir))
    with open('data/%s/tagged-batch-%d.json' % (output_dir, num_tagged_batches + 1), 'w') as f:
        json.dump(data, f)
    return True

@app.route('/post_annotation_experiment', methods=['POST'])
def post():
    output_data = request.get_json()
    ##
    crowd_data = output_data['data']
    output_dict = defaultdict(list)
    for answer in crowd_data:
        doc_id = answer['doc_key']
        output_dict[doc_id].append(answer)

    post_data_local(output_dict, pretagged=False)

    return "success"


###
# validation
###
@app.route('/render_validation_experiment')
def render_validation():
    if False:
        ## fetch data
        client = get_client()
        query = client.query(kind='newsstructure-validation-unscored')
        query.add_filter('finished', '=', False)
        results = list(query.fetch(limit=3))
    else:
        results = []
        with open('../data/news-article-flatlist/html-for-sources/doc_html.json') as f:
            for line in f.readlines():
                results.append(json.loads(line.strip()))
        import numpy as np
        results = np.random.choice(results, 3).tolist()

    num_sources = len(results)
    num_docs = len(set(map(lambda x: x['doc_id'], results)))
    return render_template('task-validation.html', sources_count=num_sources, paper_count=num_docs, input=results)

@app.route('/post_validation_experiment', methods=['POST'])
def post_validation():
    output_data = request.get_json()
    ##
    crowd_data = output_data['data']
    if False:
        client = get_client()
        query = client.query(kind='newsstructure-validation-unscored')
    else:
        json.dump(crowd_data, open('data/batch-marked-t.json', 'w'))
    return "success"


if __name__ == '__main__':
    app.run(debug=True, port=5005)