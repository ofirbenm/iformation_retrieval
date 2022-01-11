from flask import Flask, request, jsonify
from search_backend import *
from google.cloud import storage
from inverted_index_gcp1 import *
import pandas as pd


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


client = storage.Client()
# text_InvertedIndex
bucket = client.get_bucket('text_315923151')
blob = bucket.get_blob(f'postings_gcp/text_InvertedIndex.pkl')
pickle_in = blob.download_as_string()
text_InvertedIndex = pickle.loads(pickle_in)

# title_InvertedIndex
bucket = client.get_bucket('title_315923151')
blob = bucket.get_blob(f'postings_gcp/title_InvertedIndex.pkl')
pickle_in = blob.download_as_string()
title_InvertedIndex = pickle.loads(pickle_in)

# text no stem
bucket = client.get_bucket('text_nostem_315923151')
blob = bucket.get_blob(f'postings_gcp/text_nostem_InvertedIndex.pkl')
pickle_in = blob.download_as_string()
text_nostem_InvertedIndex = pickle.loads(pickle_in)

# title no stem
bucket = client.get_bucket('title_nostem_315923151')
blob = bucket.get_blob(f'postings_gcp/title_nostem_InvertedIndex.pkl')
pickle_in = blob.download_as_string()
title_nostem_InvertedIndex = pickle.loads(pickle_in)

# text bigram
bucket = client.get_bucket('text_bigram_315923151')
blob = bucket.get_blob(f'postings_gcp/text_bigram_InvertedIndex.pkl')
pickle_in = blob.download_as_string()
text_bigram_InvertedIndex = pickle.loads(pickle_in)

# title bigram
bucket = client.get_bucket('title_bigram_315923151')
blob = bucket.get_blob(f'postings_gcp/title_bigram_InvertedIndex.pkl')
pickle_in = blob.download_as_string()
title_bigram_InvertedIndex = pickle.loads(pickle_in)

# anchor
bucket = client.get_bucket('anchor_text_315923151')
blob = bucket.get_blob(f'postings_gcp/anchor_InvertedIndex.pkl')
pickle_in = blob.download_as_string()
anchor_InvertedIndex = pickle.loads(pickle_in)

# title and id for each doc
title_id = anchor_InvertedIndex.len_docs

# page views
bucket = client.get_bucket('title_315923151')
blob = bucket.get_blob('pageviews-202108-user.pkl')
pickle_in = blob.download_as_string()
views = pickle.loads(pickle_in)

# pagerank
blob = bucket.get_blob('page_rank.pkl')
pickle_in = blob.download_as_string()
pagerank_dict = pickle.loads(pickle_in)

# weights
weight = [0.97, 1.729, 0.006, 3.41, 5.2, 0.015, 0.002]
app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


# ngrop account : 23EgUdpcfTNAtFFe50VixXvhBGE_5TutdjBft5btX2wzMLUxn
@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    inverted_bucket = {
        'text': (text_InvertedIndex, 'text_315923151'),
        'title': (title_InvertedIndex, 'title_315923151'),
        'anchor': (anchor_InvertedIndex, 'anchor_text_315923151'),
        'text_bi': (text_bigram_InvertedIndex, 'text_bigram_315923151'),
        'title_bi': (title_bigram_InvertedIndex, 'title_bigram_315923151'),
        'pagerank': pagerank_dict
    }
    weight_dict = {
        'title_bm25': weight[0],
        'text_bm25': weight[1],
        'text_tf': weight[2],
        'text_bi_tf': weight[3],
        'title_bi_bm25': weight[4],
        'anchor': weight[5],
        'pagerank': weight[6]
    }
    res = predict(query, inverted_bucket, weight_dict)
    merged = [(id, title_id[id]) for id in res]
    return jsonify(merged)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    query = simple_tokenize(query)
    res = fastCosineScore(query, text_nostem_InvertedIndex, 'text_nostem_315923151', 100)
    merged = [(tup[0], title_id[tup[0]]) for tup in res]
    return jsonify(merged)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    res = binary_ranking(query, title_nostem_InvertedIndex, 'title_nostem_315923151')
    merged = [(tup[0], title_id[tup[0]]) for tup in res]
    return jsonify(merged)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    res = binary_ranking(query, anchor_InvertedIndex, 'anchor_text_315923151', token=False)
    merged = [(tup[0], title_id[tup[0]]) for tup in res]
    return jsonify(merged)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    res = pagerank_ranking(wiki_ids, pagerank_dict)
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    res = views_ranking(wiki_ids, views)
    return jsonify(res)

@app.route("/get_map")
def get_map():
    '''
        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/get_map?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    inverted_bucket = {
        'text': (text_InvertedIndex, 'text_315923151'),
        'title': (title_InvertedIndex, 'title_315923151'),
        'anchor': (anchor_InvertedIndex, 'anchor_text_315923151'),
        'text_bi': (text_bigram_InvertedIndex, 'text_bigram_315923151'),
        'title_bi': (title_bigram_InvertedIndex, 'title_bigram_315923151'),
        'pagerank': pagerank_dict
    }
    weight_dict = {
        'title_bm25': weight[0],
        'text_bm25': weight[1],
        'text_tf': weight[2],
        'text_bi_tf': weight[3],
        'title_bi_bm25': weight[4],
        'anchor': weight[5],
        'pagerank': weight[6]
    }
    res = evaluate(queries, inverted_bucket, weight_dict)
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
