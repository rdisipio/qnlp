#!/usr/bin/env python

import os, sys
import pickle

import numpy as np
import pandas as pd
import urllib.request
import re
import feedparser
import time
from tqdm import tqdm

import tensorflow_hub as hub

categories = ['astro-ph', 'cs.AI' ]
map_categories = { i:c for c,i in enumerate(categories) }

total_results = 1000
if len(sys.argv) > 1:
    total_results = int(sys.argv[1])
results_per_iteration = total_results//10 if total_results < 1000 else total_results//100
wait_time = 3 # seconds

print("There are {} known categories: {}".format(len(categories), categories))
print("Each batch will contain {} articles".format(results_per_iteration))

base_url = 'http://export.arxiv.org/api/query?'

df = pd.DataFrame(columns=["abstract", "category_txt", "category_id"])

for category in categories:
    print("Processing category {}...".format(category))
    c_id = map_categories[category]

    # Search parameters
    search_query = "cat:{}".format(category)
    start = 0

    articles_in_batch = []
    for i in tqdm(range(start, total_results, results_per_iteration)):
        query = "search_query={}&start={}&max_results={}".format(search_query, i, results_per_iteration )
        url = base_url + query
        response = urllib.request.urlopen(url)
        feed = feedparser.parse(response)
        
        for entry in feed.entries:
            abstract = entry.summary
            #clean_abstract = clean_text(abstract)
            #clean_abstract = normalize_text_nltk(clean_abstract)
            article = {
                'abstract':abstract,
                'category_txt':category,
                'category_id':c_id,
            }
            articles_in_batch.append(article)
    
    print("Found {} articles for category {}".format(len(articles_in_batch), category))
    for article in articles_in_batch:
        df = df.append(article, ignore_index=True)

    #articles = Parallel(n_jobs=-1)(delayed(_add_abstract)(i, df) for i in tqdm(range(start, total_results, results_per_iteration)))
    #print(articles)
    #

print("Found {} articles".format(df.shape[0]))
#print(df.head())
print(df.sample(n=10))

f_out = open("arxiv_abstracts.pkl", 'wb')
pickle.dump(df, f_out)
f_out.close()
