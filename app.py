
import numpy as np
import pandas as pd
import streamlit as st
import random

# for reproducibility , to get the same results when evry your run
np.random.seed(2021) 

import re
import string
from collections import Counter


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from fuzzywuzzy import fuzz


from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.svm import LinearSVC

from translate import Translator


# import nltk
# from nltk.corpus import stopwords, words
# from nltk.tokenize import word_tokenize
# from nltk import ngrams


from pprint import pprint

import sys
import os
import glob
import pickle

# here i used stop words file intate of using NLTK, because it's have big storage for deployment
file = open("english_stopwords.txt", "r")
stop_words = file.read().split()
print(stop_words) 

# stop_words = stopwords.words('english')
# stemmer = nltk.SnowballStemmer('english')

data = pd.read_csv('popular_quotes.csv')
# print(data.head())

#---------------------------------------------------------------------------------#
                        # Importing fuction for useing
#---------------------------------------------------------------------------------#

# from franco_arabic_transliterator.franco_arabic_transliterator import *
# from ar_corrector.corrector import Corrector
# from translate import Translator

# def franco_arabic(sent):
#     '''
#         Converting the Franco to text by installing this line
#         pip install franco_arabic_transliterator
#     '''
#     transliterator = FrancoArabicTransliterator() # "lexicon" OR "language-model"
#     res = transliterator.transliterate(sent, method= "lexicon") # ازيك يا حبيبي
#     return res


# def arabic_correct(sent):
#     '''
#         Arabic autocrrect from the translated franco text.
#         if the type is bool it's correct word else it a list of correct words.
#         to get the first item on list and get the first element in the tuple wihch is the correct word.
#     '''
#     corr_ans = " ".join([corr.spell_correct(word,1)[0][0] if type(corr.spell_correct(word,1)) != bool else word for word in sent.split()])
#     return corr_ans


# def arabic_to_english(sent):
#     '''
#         translate Arabic to English text by Google translation 
#     '''
#     translator = Translator(from_lang= 'ar', to_lang="en")
#     translation = translator.translate(sent)
#     return translation

def franco_to_english(sent):
    '''
        Take the franco as a input text and convert it to Arabic then English,
        and return English with correct splilling text
    '''
    # Franco to Arabic
    translator = Translator(from_lang= 'en', to_lang="ar")
    franco_arabic = translator.translate(sent)

    # Arabic to English
    translator= Translator(from_lang= 'ar', to_lang="en")
    arabic_english =  translator.translate(franco_arabic)
    return arabic_english

def clean_text(text):
    '''
        Responsible for normalize like: make it lowercase, remove text in square brackets,
        remove links, remove punctuation and remove words containing numbers.
    '''
    # text = re.findall('“([^"]*)”', text)[0] # extract text for quotations
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)  
    text = re.sub('<.*?>+', '', text) 
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # remove punctuation
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text) # url
    return text

def preprocess_data(text):
    '''
        Call the clean_text function which is responsible for text normalization, and do the other things
        for normalization like removing stop words and stemming.
    '''
    # call the fist clean text
    text = clean_text(text) 
    
    # removing stop words
    text =  ' '.join(word for word in text.split() if word not in stop_words)
    
    # stemming 
    # text = ' '.join(stemmer.stem(word) for word in text.split())
    
    return text
    

# simple predicting quotes tags
def simple_multi_label(text):
    '''
        Spliting the input text and iterave over all tags, and check if any word = tag will return it.
    '''
    text = preprocess_data(text)
    predicted_tags = []
    for word in text.split():
        if word in cleaned_tags:
            predicted_tags.append(tags_dict[word])
    return predicted_tags


# predicting quotes tags by fuzzywuzzy
def fuzzywuzzy_sim(q):
    '''
        Get the most tags similar to the quote by calculate the Levenshtein distance which is fuzzywuzzy works by it.
    '''
    tags_scores = []
    sim_tags = []
    for tag in top_20_tag:
        score = fuzz.partial_ratio(q, tag)
        tags_scores.append((score, tag))

    tags_scores.sort(reverse= True)

    for sim_score in tags_scores:
        sim_tags.append(sim_score[1])

    return sim_tags[:5]


# predicting quotes tags by ML model
def ml_predicted_tags(q):
    '''
        Get the predcted tags by the ML model
    '''
    lebels = []
    x = tf_idf.transform([q])
    labels = list(multi_label.inverse_transform(clf.predict(x))[0])
    return labels

# Combination of some function for prediction tags:
def some_pred_funcs_with_clf(q):
    '''
        Compination for predected tags by all models we made.
    '''
    l1 = ml_predicted_tags(q)
    l2 = simple_multi_label(q)
    l3 = fuzzywuzzy_sim(q)
    tags_without_modeling = list(set(l2 + l3))
    tags_with_modeling = list(set(l1+l2+l3))
    if len(tags_without_modeling) < 5:
        return tags_without_modeling
    else:
        return tags_with_modeling
    

#---------------------------------------------------------------------------------#
# when we uploading data, everything convert to str, such as here the list convert to str!
# print(type(data['tags'][0]))

# Remove ' and , from the string and [] and spliting the tags
data['tags'] = data['tags'].apply(lambda tags: tags.replace("'","").replace(",","")[1:-1].split())

# print(type(data['tags'][0]))


# Cleaning the quotes 
data['clean_text'] = data['quotes'].apply(preprocess_data)

# Get all tags in all data
tag_list = [tag for each_tag_row in data['tags'] for tag in each_tag_row]

# Knowing the frequance of tags, and sort them descending order
freq_tags = Counter(tag_list)
freq_tags_sored = sorted(freq_tags.items(), key= lambda pair: pair[1], reverse= True)
freq_tags_df = pd.DataFrame(freq_tags_sored, columns= ['word', 'counts'])

# List of unique tags
unique_tags = list(set(tag_list))
# print(len(unique_tags))

# Make the same text preprocessing/norlalization like the input text for tags also
cleaned_tags = [preprocess_data(tag) for tag in unique_tags]

# Make a dict for matching cleaning tags and original tags
# In the searchin stage, search in the clean tags and return the original one in the dic
tags_dict =  dict( [(preprocess_data(tag) , tag) for tag in unique_tags] )

# Get the 10 of most frequent tags
top_20_tag = list(freq_tags_df['word'][:20])
top_10_tag = top_20_tag[:10]
# print(top_20_tag)


# Here making new column to make  5 random tags from the 10 most frequent tags to dicrease the search space.
data['customize_top_5_tags'] = data['tags'].apply(lambda tag: random.sample(top_10_tag, 5))


#---------------------------------------------------------------------------------#

# Run  ML the
multi_label = MultiLabelBinarizer()
y = multi_label.fit_transform(data['customize_top_5_tags'])

# Spliting data
X_train, X_test, y_train, y_test = train_test_split(data['clean_text'], y, test_size= 0.2, random_state = 2021)

# TF-IDF
tf_idf = TfidfVectorizer(analyzer= 'word', max_features=10000, ngram_range=(1,3))
X_train = tf_idf.fit_transform(X_train)
X_test = tf_idf.transform(X_test)

# Run SGD classifier
model = SGDClassifier()
clf = OneVsRestClassifier(model)
clf.fit(X_train, y_train)

# testing 

# quote = "If you don't belong, don't be long!"

# print('The ML func for predicting tags:')
# print(ml_predicted_tags(quote))


# print('Combination of some function for prediction tags:')
# print(some_pred_funcs_with_clf(quote))

#-------------------------------------------------------------------------------------#
#-------------------------------Starting Streamlit App--------------------------------#


st.title("Hello World It's Goodread Tags Prediction App!")

# Initialize the variable to avoide the same result
input_quote = ""
pred_tags = []

input_quote = st.text_input('Enter Your English Or Franco-Arabic Quote Here!')

is_english = True

# Check if the input is exist not empyt string
if input_quote:
    # Checking if the input is franco if it's containe numbers
    for ch in input_quote:
        if ch.isdigit():
            # Convert franco to english
            english_text = franco_to_english(input_quote)
            clean_quote = preprocess_data(english_text)
            break

    # Make text preprocessing for the input text        
    clean_quote = preprocess_data(input_quote)
    
    # checking if it's just english words here by NLTK
    # for word in clean_quote.split():
    #     if word not in words.words():
    #         st.write("Please just enter English words!")
    #         is_english = False
    #         break
    
    # Checke if the input is English language
    for word in clean_quote.split():
        if not re.findall(r"^[a-zA-Z]+", word):
            st.write("Please just enter English words!")
            is_english = False
            break
        
    if is_english:
        # Get the predicted tags from the modeling func    
        pred_tags = list(some_pred_funcs_with_clf(clean_quote))
        st.write('### The Predicted Tags Are:')
        df = pd.DataFrame({"Predicted Tags":pred_tags})
        st.table(df.T)

        # Generate similar quotes based on the tags.
        similar_quotes = []
        # print(pred_tags)
        
        # Check if any predicted tag it's appear in the quotes list, will print the quote
        for tag in pred_tags:
            for i in range(0,11):
                if tag in data['tags'][i]:
                    similar_quotes.append(data['quotes'][i])
                    
        if st.button('Press Here For Similar Quote!'):
            st.write(f"{random.choice(similar_quotes)}")




# echo 'web: sh setup.sh && streamlit run app.py' >Procfile

# echo 'mkdir -p ~/.streamlit/
# echo "\
# [server]\n\
# headless = true\n\
# port = $PORT\n\
# enableCORS = false\n\
# \n\
# " > ~/.streamlit/config.toml' >setup.sh
