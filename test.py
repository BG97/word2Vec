# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:54:57 2020

@author: benny
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:40:36 2020

@author: benny
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
import sys
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.pipeline import make_pipeline
import operator
from sklearn import svm
from sklearn.model_selection import cross_val_score

import csv

from nltk.tag import pos_tag
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import stopwords

#st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz','stanford-ner.jar')

if len(sys.argv)>1:
    data = sys.argv[1]
else:
    data = 'Fake news/fake_or_real_news.csv'

##### MAIN #####

#open file and read data
f = open(data, newline='', encoding='utf-8')
reader = csv.reader(f, delimiter=',')
#f = open("input.csv")
#f = open("mydata.csv")
#tags=f.readlines()

i = 0
toremove = {}

#use word stems
stemmer = SnowballStemmer("english")

#fs = sys.argv[1]
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
nltk_stopwords = stopwords.words('english')
nltk_stopwords.append("'s")
##### FUNCTIONS #####
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)

    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
            
    stems = [stemmer.stem(t) for t in filtered_tokens]
    #Nostems = [t for t in filtered_tokens]
    return stems
import pandas as pd
data=[]

fake_Data=[]
real_Data=[]
for row in reader:
        #remove non-ascii from the line
    if(i == 0):
        i = 1
        continue
    line = row[2]
    line = re.sub(r"[^\x00-\x7F]+", "", line)
    line = re.sub(r"\n", "", line)
    tokenized = tokenize_and_stem(line)
#        if(re.search(r"\D",l[0]) or (l[3] != 'FAKE' and l[3] != 'REAL')):
#            continue
    if(row[3] == 'FAKE'):
        fake_Data = fake_Data+tokenized
    else:
        real_Data = real_Data +tokenized
from gensim.models import Word2Vec
import gensim
from keras.models import load_model
if len(sys.argv)>1:
    model_fake = Word2Vec.load('Fake_'+sys.argv[2])
    model_real = Word2Vec.load('Real_'+sys.argv[2])
else:
    model_fake = Word2Vec.load('Fake_model.model')
    model_real = Word2Vec.load('Real_model.model')


all_stopwords = gensim.parsing.preprocessing.STOPWORDS

import numpy as np

def cosine_distance (model, word,target_list , num) :
    cosine_dict ={}
    word_list = []
    a = model[word]
    for item in target_list :
        if item != word :
            b = model [item]
            cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
            cosine_dict[item] = cos_sim
    dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order 
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    noCommon=[]
    
    for i,word in enumerate(word_list): 
        if len(noCommon)<5:
            if not word[0] in nltk_stopwords and not word[0] in all_stopwords:
                noCommon.append(word)
                
        
    return noCommon
wordList=[]
if len(sys.argv)>1:  
    f = open(sys.argv[3], newline='', encoding='utf-8')
    reader = csv.reader(f, delimiter=' ')
    for word in reader:
        wordList.append(word[0])
else:
    f = open('query.txt', newline='', encoding='utf-8')
    reader = csv.reader(f, delimiter=' ')
    for word in reader:
        wordList.append(word[0])

output='Fake news top 5 stemmed words\n'
for i,val in enumerate(wordList):
    val1=val.lower()
    output_fake = cosine_distance(model_fake,stemmer.stem(val1),fake_Data,5)
    fake_top5 = []
    for j in range(len(output_fake)):
        fake_top5.append(output_fake[j][0])
    output=output+'The most similar word for '+val+' is '+str(fake_top5)+'\n'
print(output)



output='Real news top 5 stemmed words'

for i,val in enumerate(wordList):
    val1=val.lower()
    output_real = cosine_distance(model_real,stemmer.stem(val1),real_Data,5) 
    real_top5 = []
    for j in range(len(output_real)):
        real_top5.append(output_real[j][0])
    output=output+'The most similar word for '+val+' is '+str(real_top5)+'\n'
print(output)















