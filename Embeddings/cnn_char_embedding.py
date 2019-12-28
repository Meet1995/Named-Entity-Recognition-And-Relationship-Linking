# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:08:49 2019

@author: mg21929
"""
import pandas as pd
import numpy as np
from contractions import CONTRACTION_MAP
import unicodedata
import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from pathlib import Path 

ADE_DATA_PATH = Path(r'C:\Users\mg21929\Documents\Case Study\ADE-Corpus-V2')


nlp = spacy.load('en_core_web_md', parse=True, tag=True, entity=True)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
temp_list=[]
for value in stopword_list:
    temp_list.append(value.capitalize())
for value in temp_list:
    stopword_list.append(value)
    
    


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
# expand_contractions("Y'all can't expand contractions I'd think")

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, ' ', text)
    return text
# remove_special_characters("Well this was fun! What do you think? %123#@!", remove_digits=False)

def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text
# simple_stemmer("My system keeps crashing his crashed yesterday, ours crashes daily")

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text
# lemmatize_text("My system keeps crashing! his crashed yesterday, ours crashes daily")

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
#    if is_lower_case:
    filtered_tokens = [token for token in tokens if token not in stopword_list]
#    else:
#        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
# remove_stopwords("The, and, if AIN stopwords, computer is not")

def remove_single_words(text):
    word_list= text.split(" ")
    word_list= [i for i in word_list if len(i) > 1]
    text = ' '.join(words for words in word_list)
    return text
    
def normalize_corpus(corpus, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_lemmatization=True, special_char_removal=True, 
                     stopword_removal=True, remove_digits=True, single_words_removal=True):
    
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        # expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
        # lowercase the text    
        if text_lower_case:
            doc = doc.lower()
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
        # remove special characters and\or digits    
        if special_char_removal:
            # insert spaces between special characters to isolate them    
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)  
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
        # Remove single words
        if single_words_removal:
            doc = remove_single_words(doc)
            
        normalized_corpus.append(doc)
        
    return normalized_corpus


def vec(char):
    chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    chars=list(chars) 
    arr=np.zeros((52,1))
    for i in range(len(chars)):
        if char == chars[i]:
            arr[i]=1
    return arr

def DataGen(word):
    data_x=[]
    data_y=[]
    word_chars=list(word) 
    for i in range(len(word_chars)):
        mat=np.zeros((52,3))
        data_y.append(vec(word_chars[i]))
        if i==0:
            mat[:,1:2]=vec(word_chars[i])
            mat[:,2:3]=vec(word_chars[i+1])
        if i==len(word_chars)-1:
            mat[:,0:1]=vec(word_chars[i-1])
            mat[:,1:2]=vec(word_chars[i])  
        if i not in [0,len(word_chars)-1]:
            mat[:,0:1]=vec(word_chars[i-1])
            mat[:,1:2]=vec(word_chars[i])
            mat[:,2:3]=vec(word_chars[i+1])
        data_x.append(mat)
    data_x=np.asarray(data_x, dtype=float)
    data_y=np.asarray(data_y, dtype=float)
    return data_x,data_y

def TrainDatGenerator(corpus):
    data_list=[] 
    lable_list=[]      
    for sentence in corpus:
        word_list= sentence.split(" ")
        x=[]
        y=[]
        for words in word_list:
            char_images,char_lables = DataGen(words)
            x.append(char_images)
            y.append(char_lables)
        sent_data= x[0]
        lable_data= y[0]
        for i in range(1,len(x)):
            sent_data=np.concatenate((sent_data, x[i]), axis=0)
            lable_data=np.concatenate((lable_data, y[i]), axis=0)
        data_list.append(sent_data)
        lable_list.append(lable_data)
        
    data=data_list[0]
    lable_data_compiled=lable_list[0]

    for i in range(1,len(data_list)):
        data=np.concatenate((data, data_list[i]), axis=0)
        lable_data_compiled=np.concatenate((lable_data_compiled, lable_list[i]), axis=0)
    
    return data[:,:,:,np.newaxis], np.squeeze(lable_data_compiled, axis=2)

# =============================================================================
# 
# =============================================================================
    
drug_ae = pd.read_csv(str(ADE_DATA_PATH/"DRUG-AE.rel"), sep = '|', header=None)

drug_ae.columns = ['PubMed_ID', 'Sentence', 'Adverse_effect', 'Begin_offset_of_Adverse_Effect_at_document_level',
             'End_offset_of_adverse_effect_at_document_level', 'Drug', 'Begin_offset_of_drug_at_document_level',
             'End_offset_of_dug_at_document_level'] 
        
corpus = normalize_corpus(drug_ae['Sentence'], remove_digits=True, text_lower_case=False, text_lemmatization=False, 
                          special_char_removal=True, single_words_removal=True)

X_train, y_train = TrainDatGenerator(corpus)

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense

C= 3
n_dim= 52

# Initialising the CNN
encoder = Sequential()

encoder.add(Conv2D(1, (1,C), strides=(1,1), input_shape= (n_dim,C,1), activation= 'tanh'))

encoder.add(Flatten())

encoder.add(Dense(units = 52, activation = 'relu'))


# Compiling the CNN
encoder.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

encoder.fit(X_train, y_train, epochs = 1)

# =============================================================================
# =============================================================================

test =['House', 'mouse']
x1,y1=DataGen(test[0])
x2,y1=DataGen(test[1])

x1=x1[:,:,:,np.newaxis]
x2=x2[:,:,:,np.newaxis]

a1=encoder.predict(x1)
a2=encoder.predict(x2)

char_embedding_1=[]
char_embedding_2=[]

for i in range(52):
    char_embedding_1.append(max(a1[:,i]))
    char_embedding_2.append(max(a2[:,i]))
    
from scipy.stats.stats import pearsonr   
pearsonr(char_embedding_1,char_embedding_2)

model_json = encoder.to_json()
with open("encoder.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
encoder.save_weights("encoder.h5")
print("Saved model to disk")
