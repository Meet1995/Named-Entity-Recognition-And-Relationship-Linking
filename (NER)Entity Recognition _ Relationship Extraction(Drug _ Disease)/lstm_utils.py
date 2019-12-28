# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:02:37 2019

@author: mg21929
"""

import os
import random
from keras.models import model_from_json
import pandas as pd
import numpy as np
from contractions import CONTRACTION_MAP
import unicodedata
import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from pathlib import Path 
import gensim
from collections import Counter
import operator
import networkx as nx
from keras.preprocessing.text import Tokenizer 
from sklearn.decomposition import PCA
import keras.backend as K


os.environ['KERAS_BACKEND' ] = 'tensorflow' ## Use tensorflow or theano
ADE_DATA_PATH = Path(r'C:\Users\mg21929\Documents\Case Study\ADE-Corpus-V2')


nlp = spacy.load('en_core_web_md', parse=True, tag=True, entity=True)    

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
    stopword_list = nltk.corpus.stopwords.words('english')
    temp_list=[]
    for value in stopword_list:
        temp_list.append(value.capitalize())
    for value in temp_list:
        stopword_list.append(value)
    stopword_list.append("cannot")  
    tokenizer = ToktokTokenizer()
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

def get_char_embeddings(corpus):
    combined_data=corpus[0]
    for i in range(1,len(corpus)):
        combined_data =combined_data + " " + corpus[i]
    
    word_list= combined_data.split(" ")
    # load json and create model
    json_file = open('encoder.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    encoder = model_from_json(loaded_model_json)
    # load weights into new model
    encoder.load_weights("encoder.h5")
    print("Loaded model from disk")        
    char_embedding_list=[]
    for word in word_list:
        char_embedding=[]
        x1,y1=DataGen(word)
        x1=x1[:,:,:,np.newaxis]
        a1=encoder.predict(x1)
        for i in range(52):
            char_embedding.append(max(a1[:,i]))
        char_embedding_list.append(char_embedding)
    return char_embedding_list
    
def get_pos_embeddings(corpus):
    sent_pos_tags=[]
    for i in range(len(corpus)):
        sent_pos_tags.append([word.pos_ for word in nlp(corpus[i])])
    word_pos_tags=sent_pos_tags[0]
    for i in range(1,len(sent_pos_tags)):
        word_pos_tags = word_pos_tags + sent_pos_tags[i]
    
    embeddings= []
    for tag in word_pos_tags:
        pos_tag_dict = {'DET':0, 'ADJ':0, 'VERB':0, 'PRON':0, 'NOUN':0, 'CCONJ':0, 
                'ADV':0, 'INTJ':0, 'ADP':0, 'PART':0, 'PROPN':0, 'PUNCT':0, 'NUM':0, 'X':0
               }
        pos_tag_dict[tag] = 1
        embeddings.append(list(pos_tag_dict.values()))
    return embeddings

def get_word_embeddings(corpus):
    """
    word2vec incrementally trained on the corpus
    """
    data= []
    for sent in corpus:
        data.append(sent.split(" ")) 
    
    model_new = gensim.models.Word2Vec(min_count=1,size=200,window=4, iter = 50)# iter == epochs
    model_new.build_vocab(data)
    model_new.intersect_word2vec_format(str(ADE_DATA_PATH/'pubmed2018_w2v_200D.bin'),binary=True,lockf=0.0)
    model_new.train(data,total_words=model_new.corpus_count,epochs=model_new.epochs)
    
    embeddings= []
    word_list= []
    for lists in data:
        word_list= word_list + lists
    for word in word_list:
        embeddings.append(model_new[word])
    return embeddings

def get_labels(df_row):
    """
    --------------
    Input: Row in a dataframe
    Output: BILOU tags for every word in sentene
    
    """
    sent=df_row['docs'].lower()
    tokens=re.split(' |-',sent)
    labels=['O']*len(tokens)
    disease=re.split(' |-',df_row['adverse_effect'])
    drug=re.split(' |-',df_row['drug'])
    label_mapping={1:['U'],2:['B','L']}
    
    for i in range(3,25):
        label_mapping[i]=['B']+['I']*(i-2)+['L']
        
    bilou_dis=label_mapping[len(disease)]
    bilou_drug=label_mapping[len(drug)]
    
    try:
        for i,tag in enumerate(bilou_dis):
            pos = tokens.index(disease[i].lower())
            labels[pos]= tag+'-Disease'
    except:
        pass
    try:
        
        for i,tag in enumerate(bilou_drug):
            pos = tokens.index(drug[i].lower())
            labels[pos]=tag+'-Drug'
    except:
        pass
    return labels

def get_ohe_lables(corpus, df_drug, df_effect):
    clean_df = pd.concat([pd.DataFrame(corpus, columns = ['docs']), df_drug, df_effect], axis = 1)
    lables_list = []
    lables_list.append(clean_df.apply(lambda x: get_labels(x),axis=1))
    all_word_lables = []
    for i in range(len(lables_list[0])):
        all_word_lables = all_word_lables + lables_list[0][i]
        
    # OHE for lables
    df_all_word_lables= pd.DataFrame(all_word_lables, columns = ["lables"])
    df_all_word_lables_ohe = pd.get_dummies(df_all_word_lables['lables'])
    all_word_lables_ohe = []
    for i in range(len(df_all_word_lables_ohe)):
        all_word_lables_ohe.append(list(df_all_word_lables_ohe.iloc[i,:]))
        
    lable_map={}
    for i in range(len(df_all_word_lables_ohe.columns)):
        lable_map.update({df_all_word_lables_ohe.columns[i] : i})   
        
    return all_word_lables_ohe, all_word_lables, lable_map

def lstm_data_generator(stride, step_size, char_embeddings, pos_embeddings, word_embeddings, all_word_lables_ohe):
    all_embeddings = []
    for i in range(len(char_embeddings)):
        all_embeddings.append(char_embeddings[i] + list(word_embeddings[i]) + pos_embeddings[i])    
        
    y = []
    X = []
    for i in range(0, len(all_embeddings)-step_size+1, stride):
        X.append(all_embeddings[i : i + step_size])
        y.append(all_word_lables_ohe[i : i + step_size])                
    return np.array(X), np.array(y)

def get_lable_from_ohe(y_ohe):
    y_lable = []
    for i in range(len(y_ohe)):
            k = ([np.argmax(l) for l in list(y_ohe[i])])
            y_lable.append(k)
            k=[]
    return y_lable
   
def unroll_data(y_data):
    unrolled_data=[]
    temp_array = np.array(y_data)
    diags = [temp_array[::-1,:].diagonal(i) for i in range(-temp_array.shape[0]+1,temp_array.shape[1])]
    diags.extend(temp_array.diagonal(i) for i in range(temp_array.shape[1]-1,-temp_array.shape[0],-1))
    k = [n.tolist() for n in diags]
    k = k[:int(len(k)/2)]
    for i in range(len(k)):
        m = dict(Counter(k[i]))
        unrolled_data.append(max(m.items(), key=operator.itemgetter(1))[0])
    return unrolled_data

def unroll_hidden_features(h_X):
    hidden_features=[]
    temp_array = np.array(h_X)
    
    diags = [temp_array[::-1,:].diagonal(i) for i in range(-temp_array.shape[0]+1,temp_array.shape[1])]
    diags.extend(temp_array.diagonal(i) for i in range(temp_array.shape[1]-1,temp_array.shape[0],-1))
    k = [n.tolist() for n in diags]
    k = k[:len(k)]
    for i in range(len(k)):
        avg_h = np.mean(np.array(k[i]), axis = 1)
        hidden_features.append(avg_h)
    return hidden_features

def SDP_btw_Words(sentence, source_word, target_word):
    doc = nlp(sentence)
    relations = {}
    token_map = {}
#    spacy.displacy.serve(doc, style='dep')
    edges = []
    for token in doc:
        token_map.update({token.text : token.i})
        for child in token.children:
            edges.append(('{0}'.format(token.text), '{0}'.format(child.text)))
            relations.update({token.text +'-' + child.text : child.dep_})
    
    lca_matrix = doc.get_lca_matrix()
#    uniques = []
#    for i in range(len(doc)):
#        uniques = uniques +  list(np.unique(lca_matrix[:,i]))
#    root_word = doc[max(set(uniques), key = uniques.count)].text
    
    graph_directed = nx.DiGraph(edges)
    graph_undirected = nx.Graph(edges)

    index_lca = lca_matrix[token_map[source_word],token_map[target_word]]
    for key in token_map:
        if token_map[key]==index_lca:
            lca1 = key

    sdp = nx.shortest_path(graph_undirected, source=source_word, target=target_word)
    
    lca2 = nx.lowest_common_ancestor(graph_directed, source_word, target_word)
    
    if lca1 in sdp:
        lca = lca1
    else:
        lca = lca2
    
    sdp_dep = []
    temp = []
    for i in range(len(sdp) -1):
        if i == 0:
            key1 = sdp[i] + '-' + sdp[i+1]    
            key2 = sdp[i+1] + '-' + sdp[i]
            
            if key1 in relations.keys():
                temp = [sdp[i] , relations[key1] , sdp[i+1]]
            else:
                temp = [sdp[i] , relations[key2] , sdp[i+1]]
        else:
            key1 = sdp[i] + '-' + sdp[i+1]    
            key2 = sdp[i+1] + '-' + sdp[i]
            
            if key1 in relations.keys():
                temp = [relations[key1] , sdp[i+1]]
            else:
                temp = [relations[key2] , sdp[i+1]]        
        sdp_dep = sdp_dep + temp
    
    for i in range(len(sdp_dep)):
        if sdp_dep[i]==lca:
            a_k = sdp_dep[0:i+1]
            k_a = a_k[::-1]
            k_b = sdp_dep[i:]
            b_k = k_b[::-1]
 
    return  a_k, k_a, k_b, b_k, sdp, sdp_dep, lca     
#SDP_btw_Words('insulin allergy may complete resolution symptoms standard desensitization particularly patients concomitant protamine allergy','insulin','allergy')

def get_lable_indices_by_sentences(corpus, predicted_lables):
    word_list = []
    ini = 0
    entity_indices = []
    entity_words_sent = []
    for i in range(len(corpus)):
        len_sent = len(corpus[i].split(" "))
        word_list = word_list + corpus[i].split(" ")
        sent_lables = predicted_lables[ini:ini+len_sent]
        temp = []
        temp_word = []
        for j in range(len(sent_lables)):
            if sent_lables[j] in [2,3,4,5,7,8]:
                temp.append(j+ini)
                temp_word.append(word_list[j+ini])
        entity_indices.append(temp) 
        entity_words_sent.append(temp_word)
        ini=len_sent + ini   
        
    def tag_cleaner(ll):
        temp = []
        for i in range(len(ll)-1):
            if ll[i+1]-ll[i] == 1:
                temp.append(ll[i+1])
            else:
                temp.append(ll[i])
            temp.append(ll[-1])    
            temp = list(np.unique(temp))
        return temp

    cleaned_entity_indices = []   
    for i in range(len(entity_indices)):
        cleaned_entity_indices.append(tag_cleaner(entity_indices[i]))
    
    return entity_indices, cleaned_entity_indices, word_list, entity_words_sent

def get_sdp_and_relation_lables(df_drug, df_effect, cleaned_entity_indices, corpus, word_list, h_X_unrolled):
    from itertools import combinations as cmbs
    d_a_k = []
    h_a = []
    h_b = []
    d_k_a = []
    d_k_b = []
    d_b_k = []
    rel_lables = []

    entity_list =  list(set(df_drug['drug'].values.tolist() + df_effect['adverse_effect'].values.tolist()))
    for i in range(len(cleaned_entity_indices)):
        if len(cleaned_entity_indices[i])>=2:
            comb = list(cmbs(cleaned_entity_indices[i], 2))
            for combinations in comb:
                try:
                    temp = SDP_btw_Words(corpus[i], word_list[combinations[0]], word_list[combinations[1]])
                    d_a_k.append(temp[0])
                    h_a.append(h_X_unrolled[combinations[0]])
                    h_b.append(h_X_unrolled[combinations[1]])
                    d_k_a.append(temp[1])
                    d_k_b.append(temp[2])
                    d_b_k.append(temp[3])
                    
                    if word_list[combinations[0]] in entity_list and word_list[combinations[1]] in entity_list:
                        rel_lables.append(1)
                    else:
                        rel_lables.append(0)
                except:
                    pass
    return d_a_k, d_k_a, d_k_b, d_b_k, np.matrix(h_a), np.matrix(h_b), np.matrix(rel_lables).T

def get_sdp_encoding(d_a_k, d_k_a, d_k_b, d_b_k, apply_pca = True, n_component = 990):      
    sdp_corpus=[]    
    for values in d_a_k+d_b_k:
            sdp_corpus = sdp_corpus + values
    unique_sdp_corpus = list(set(sdp_corpus))
    
    d_a_k_ohe = []
    d_k_a_ohe = []
    d_k_b_ohe = []
    d_b_k_ohe = []
    tokenizer_sdp = Tokenizer(num_words=len(unique_sdp_corpus))
    tokenizer_sdp.fit_on_texts(unique_sdp_corpus)
    for i in range(len(d_a_k)):
        d_a_k_ohe.append(np.mean(np.array(tokenizer_sdp.texts_to_matrix(d_a_k[i], mode='binary')), axis=0))
        d_k_a_ohe.append(np.mean(np.array(tokenizer_sdp.texts_to_matrix(d_k_a[i], mode='binary')), axis=0))
        d_k_b_ohe.append(np.mean(np.array(tokenizer_sdp.texts_to_matrix(d_k_b[i], mode='binary')), axis=0))
        d_b_k_ohe.append(np.mean(np.array(tokenizer_sdp.texts_to_matrix(d_b_k[i], mode='binary')), axis=0))
        
    d_a_k_ohe = np.matrix(d_a_k_ohe)
    d_k_a_ohe = np.matrix(d_k_a_ohe)
    d_k_b_ohe = np.matrix(d_k_b_ohe)
    d_b_k_ohe = np.matrix(d_b_k_ohe)
    
    if apply_pca:
        # Applying PCA
        pca = PCA(n_components = n_component)
        d_a_k_ohe = pca.fit_transform(d_a_k_ohe)
        explained_variance = sum(pca.explained_variance_ratio_)
        print("Explained Variance d_a_k : ", explained_variance)
        
        d_k_a_ohe = pca.fit_transform(d_k_a_ohe)
        explained_variance = sum(pca.explained_variance_ratio_)
        print("Explained Variance d_k_a: ", explained_variance)
        
        d_k_b_ohe = pca.fit_transform(d_k_b_ohe)
        explained_variance = sum(pca.explained_variance_ratio_)
        print("Explained Variance d_k_b: ", explained_variance)
        
        d_b_k_ohe = pca.fit_transform(d_b_k_ohe)
        explained_variance = sum(pca.explained_variance_ratio_)
        print("Explained Variance d_b_k: ", explained_variance)

    return d_a_k_ohe, d_k_a_ohe, d_k_b_ohe, d_b_k_ohe

def relation_lstm_data_generator(d_a_k_ohe_pca, d_k_a_ohe_pca, d_k_b_ohe_pca, d_b_k_ohe_pca, h_a, h_b, rel_lables):
    data_ac = np.concatenate((d_a_k_ohe_pca, h_a), axis=1)
    data_ca = np.concatenate((d_k_a_ohe_pca, h_a), axis=1)
    data_cb = np.concatenate((d_k_b_ohe_pca, h_b), axis=1)
    data_bc = np.concatenate((d_b_k_ohe_pca, h_b), axis=1)
    
    data_ac = data_ac[:,np.newaxis,:]
    data_ca = data_ca[:,np.newaxis,:]
    data_cb = data_cb[:,np.newaxis,:]
    data_bc = data_bc[:,np.newaxis,:]
    
    rel_lables = rel_lables[:,np.newaxis,:]
    return data_ac, data_ca, data_cb, data_bc, rel_lables

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def K_Fold(array, n_splits = 3):
    fold_size =int(len(array)/n_splits)
    lst = list(range(0,len(array)))
    random.shuffle(lst)
    split_indices = []
    k = 0  
    for i in range(n_splits):
        if i<n_splits-1:
            split_indices.append(lst[k:k+fold_size])
            k = k+fold_size
        else:
            split_indices.append(lst[k:])
    return split_indices, lst
