### sentiment analysis on amazon product review data
## TODO: extract top key words based on TFIDF score from review text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np

## python dependencies
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt


def sort_cooc(cooc_mat):
    tuples=zip(cooc_mat.col, cooc_mat.data)
    res= sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

## extract top key words from review text

def get_topn_keywords(feature_names, sorted_items, top_n=2):
    sorted_items=sorted_items[:top_n]
    score_value, feature_values, output=[],[], {}
    for idx, score in sorted_items:
        feat_name=feature_names[idx]
        score_value.append(score)
        feature_values.append(feature_names[idx])
    
    for row in range(len(feature_values)):
        output[feature_values[idx]]=score_value[idx]
    return output

## extract top key words from review text based on TF-IDF score

def get_topn_keywords_tfidf(reviews):
    key_words_arr=[]
    cv=CountVectorizer(max_df=0.85, stop_words=None)
    tfidf_trans = TfidfTransformer(use_idf=True, smooth_idf=True)
    word_count_vec = cv.fit_transform(reviews)
    tfidf_trans.fit(word_count_vec)
    feat_names=cv.get_feature_names()

    for i in range(0, len(reviews)):
        tfidf_vec=tfidf_trans.transform(cv.transform([reviews[i]]))
        sorted_vec = sort_cooc(tfidf_vec.tocoo())
        res=get_topn_keywords(feat_names, sorted_vec, top_n=2)
        key_words_arr.append(res)
    return key_words_arr


## show top key words for product review text
def get_keywords(reviews):
    key_words=get_topn_keywords_tfidf(reviews)
    sentences=[]
    for i in range(0, len(key_words)):
        sent=''.join(list(key_words[i].keys()))
        sentences.append(sent)
    return sentences

## data prep

data=pd.read_csv("amazon_prod_dataset\Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")
sent_data = data[['reviews.rating', 'reviews.text', 'reviews.title']]