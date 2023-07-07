from bs4 import BeautifulSoup
import requests
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import pickle
import csv

def summarize(text, per):
    nlp = spacy.load('en_core_web_sm')
    doc= nlp(text)
    tokens=[token.text for token in doc]
    word_frequencies={}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency=max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency
    sentence_tokens= [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
    select_length=int(len(sentence_tokens)*per)
    summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
    final_summary=[word.text for word in summary]
    summary=''.join(final_summary)
    return summary


html_text= requests.get('https://www.livemint.com/market/stock-market-news/adani-enterprises-news-live-updates-fpo-subscription-share-price-11675049205070.html').text
# print(html_text)

soup = BeautifulSoup(html_text,'lxml')
news = soup.find_all('div', class_='liveSec')
adani=[]
final=[]
newsInfo=[]
# summary=[]
translation = {39: None}
for new in  news:
    newsInfo=[]
    time = new.find('span',class_='timeStamp').text
    headline = new.find('h2', class_='liveTitle').text
    main_news = new.find_all('p')

    # print(time)
    # print(headline)
    for mainNew in main_news:
        # if mainNew == "<p></p>":
            # continue
        all_news = mainNew.get_text()
        if all_news=='':
            continue
        # print(all_news)

        
        newsInfo.append(all_news)
        # summary = summarize(newsInfo)
    newsInfo = str(newsInfo)[1:-1].translate(translation)
    # newsInfo = str(newsInfo)[1:-1]
    # print("news: ", len(newsInfo))
    # print(type(newsInfo))
    if len(newsInfo)!=0:
        summary = summarize(newsInfo, 0.5)
        if len(summary)!=0:
            analysis= TextBlob(summary)
        else:
            summary = newsInfo
            analysis= TextBlob(newsInfo)
        polarityCheck= analysis.polarity

    else:
        summary= ''
        analysis= TextBlob(headline)
        polarityCheck= analysis.polarity

    # print("sum: ", len(summary))
    analysisHeadline = TextBlob(headline)
    polarityHeadline = analysisHeadline.polarity

    if polarityCheck>0:
        responseSumm = 'Positive'
    elif polarityCheck==0:
        responseSumm = 'Neutral'
    else:
        responseSumm = 'Negative'
    
    if polarityHeadline>0:
        responseHead = 'Positive'
    elif polarityHeadline==0:
        responseHead = 'Neutral'
    else:
        responseHead = 'Negative'   
    # summary = summarize(newsInfo)
    # newsInfo = ''.join(str(newsInfo)[1:-1].translate(translation).split(','))


    # newsinfo = [*set(newsInfo)]
    # print(newsInfo)
    adani.append([headline, newsInfo, summary, polarityCheck, polarityHeadline , responseSumm, responseHead, time])
    final.append([headline,newsInfo,polarityCheck, polarityHeadline])
# print(adani)
# print(news)

df = pd.DataFrame(adani, columns= ['Headlines', 'Description' , 'Summary', 'Summary_Polarity', 'Headline_polarity','Response_Summary','Response_Headline','Timestamp'])
df.to_csv('AnalyzingStocks.csv')

datasets = pd.DataFrame(final, columns=['Headlines', 'Description','Summary_Polarity', 'Headline_polarity'])
datasets.to_csv('Stockfin.csv')
