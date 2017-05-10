# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 21:53:32 2017

@author: ghat
"""
#import nltk
#import re
#from nltk.stem.wordnet import WordNetLemmatizer
#
#index = open('C:\\Users\\ghat\\Desktop\\queryclick40000picnotnull.txt','r').readlines() #每张图片的非空query 格式为：索引 点击次数
#query_name= open('C:\\Users\\ghat\\Desktop\\40000pic\\query40000pic.txt','r').readlines()  #query的名称
#word_top8840 = open('C:\\Users\\ghat\\Desktop\\1000word.txt','r').readlines()  #词的名称
#dogpic_word=open('C:\\Users\\ghat\\Desktop\\dogpic_word1000.txt','w')
#lem = WordNetLemmatizer()
#
#for line in range(0,len(word_top8840)):
#    word_top8840[line] = word_top8840[line].split(' ')[0]
#
#for i in range(0,len(index)):
#    pindex = index[i].split(' ')
#    j=0
#    while j<(len(pindex)-1):
#        if int(pindex[j])-1 >333683:
#            j=j+2
#            continue
#        tempindex = int(pindex[j])-1
#        query_temp = query_name[tempindex]
#        word = nltk.word_tokenize(query_temp)
#        for k in range(0,int(pindex[j+1])):
#            for l in range(0,len(word)):
#                match = re.search('^[a-zA-Z0-9]+$', word[l]) 
#                if not match:
#                    continue
#                pword = lem.lemmatize(word[l])
#                if str(pword) in word_top8840:
#                    dogpic_word.write(str(pword)+' ')
#                else:
#                    continue
#        j = j+2
#    dogpic_word.write('\n')
#       # print i
#        
#dogpic_word.close()
        
######################################################华丽的分割线############################################################
        
dogpic_word=open('C:\\Users\\ghat\\Desktop\\dogpic_word1000.txt','r').readlines()
#word_top8840 = open('e:\dog_word_top8840.txt','r').readlines()
#for line in range(0,len(word_top8840)):
#    word_top8840[line] = word_top8840[line].strip(' \n')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
vectorizer = CountVectorizer(min_df=0, token_pattern=r"\b\w+\b")
transformer=TfidfTransformer()
#corpus = [    '12 12 white white dog dog',
#     'cat dog mouse fuck',
#     'ha ha white',
#     '12 cat ',
# ]
# 
x = vectorizer.fit_transform(dogpic_word)
tfidf=transformer.fit_transform(x)
#y= vectorizer.get_feature_names()
#for line in range(0,len(y)):
#    y[line] = str(y[line])
#for i in range(0,1000):
#    if word_top8840[i] not in y:
#        print word_top8840[i]
#    else:
#        continue


np.savetxt("C:\\Users\\ghat\\Desktop\\wordweight1000.txt",tfidf.toarray(),fmt="%.8lf",delimiter=" ")