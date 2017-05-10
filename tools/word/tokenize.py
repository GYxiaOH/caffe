# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:07:25 2017

@author: ghat
"""
import re
import nltk
#import nltk.stem
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

query_name = open('C:\Users\ghat\Desktop\query40000pic.txt','r').readlines()
query_clickcout = open('C:\Users\ghat\Desktop\queryclick40000pic.txt','r').readlines()
ending_file = open('C:\Users\ghat\Desktop\word40000pic.txt','w')
english_stopwords = open('E:\stopwords\stopwords.txt','r').readlines()
for line in range(0,len(english_stopwords)):
    english_stopwords[line] = english_stopwords[line].strip('\n')
endding = {}
#s=nltk.stem.SnowballStemmer('english')
lem = WordNetLemmatizer()
#english_stopwords = stopwords.words('english')
for i in range(0,333683):
    word = nltk.word_tokenize(query_name[i])
    for j in range(0,len(word)):
        match = re.search('^[a-zA-Z0-9]+$', word[j]) 
        if match:
            pword = str(lem.lemmatize(word[j]))
            if pword not in english_stopwords:
                if pword not in endding:
                    endding[str(pword)] =int(query_clickcout[j])
                else:
                    endding[str(pword)] = endding[str(pword)] +int(query_clickcout[j])
            else:
                continue
        else:
            continue
e = sorted(endding.items(),key = lambda item:item[1],reverse=True)

for c in range(0,len(endding)):
    ending_file.write(e[c][0]+' ')
    ending_file.write(str(e[c][1]))
    ending_file.write('\n')

ending_file.close();
