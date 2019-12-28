# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 2019

@author: mg21929
"""
import lstm_utils
import os
import copy
import pandas as pd
import numpy as np
from pathlib import Path 
from collections import Counter
from sklearn.metrics import  classification_report
from keras.models import Sequential, Model, model_from_json
from keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, concatenate, Input
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

#from keras.callbacks import TensorBoard

os.environ['KERAS_BACKEND' ] = 'tensorflow' ## Use tensorflow or theano
ADE_DATA_PATH = Path(r'C:\Users\mg21929\Documents\Case Study\ADE-Corpus-V2')

drug_ae = pd.read_csv(str(ADE_DATA_PATH/"DRUG-AE.rel"), sep = '|', header=None)

drug_ae.columns = ['PubMed_ID', 'Sentence', 'Adverse_effect', 'Begin_offset_of_Adverse_Effect_at_document_level',
             'End_offset_of_adverse_effect_at_document_level', 'Drug', 'Begin_offset_of_drug_at_document_level',
             'End_offset_of_dug_at_document_level'] 
        
corpus = lstm_utils.normalize_corpus(drug_ae['Sentence'], remove_digits=True, text_lower_case=False, 
                          text_lemmatization=False, special_char_removal=True, single_words_removal=True)

df_drug = pd.DataFrame(lstm_utils.normalize_corpus(drug_ae['Drug'], remove_digits=True,
                                        text_lower_case=False, text_lemmatization=False, special_char_removal=True, 
                                        single_words_removal=True), columns = ['drug'])

df_effect = pd.DataFrame(lstm_utils.normalize_corpus(drug_ae['Adverse_effect'], remove_digits=True,
                                         text_lower_case=False, text_lemmatization=False, special_char_removal=True, 
                                                           single_words_removal=True), columns =['adverse_effect'])

all_word_lables_ohe, all_word_lables, lable_map = lstm_utils.get_ohe_lables(corpus, df_drug, df_effect)

char_embeddings = lstm_utils.get_char_embeddings(corpus)

pos_embeddings = lstm_utils.get_pos_embeddings(corpus)

word_embeddings = lstm_utils.get_word_embeddings(corpus)

X , y = lstm_utils.lstm_data_generator(1,5, char_embeddings, pos_embeddings, word_embeddings, all_word_lables_ohe)

X_train = X[0: 60000]
X_test = X[60000: ]

y_train = y[0: 60000]
y_test = y[60000: ]

class_distribution = Counter(all_word_lables)
weights = {}
for key in class_distribution:
    weights.update({key : float((class_distribution[key]/72900)**-1)})
        

class_weights={}
for key in weights:
    class_weights.update({lable_map[key] : weights[key]})
    
# =============================================================================
#      LSTM for Entity Recognition
# =============================================================================
    
data_dim = 266
timesteps = 5
num_classes = 9

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(Bidirectional(LSTM(300, return_sequences=True, input_shape=(timesteps, data_dim)), name = "l1"))  
model.add(Bidirectional(LSTM(300, return_sequences=True), name = "l2"))  
model.add(Bidirectional(LSTM(300, return_sequences=True), name = "l3"))  
model.add(TimeDistributed(Dense(num_classes, activation='softmax', name = "dense"))) # OR 
# model.add(Dense(num_classes, activation='softmax', name = "dense"))( TimeDistributed or Dense both are the same here)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#tensorboard = TensorBoard(log_dir='./logs')
model.fit(X_train, y_train, batch_size=64, epochs=7)
#model.fit(X, y, batch_size=64, epochs=1, callbacks=[tensorboard], sample_weight=class_weights)

split_indices , lst = lstm_utils.K_Fold(X_train, n_splits = 5) 

score = []
f1_score_cross_val_entity = []
for i in range(len(split_indices)):
    temp = copy.deepcopy(lst)
    for value in split_indices[i]:
        temp.remove(value)
    x_trn, x_val = X_train[temp], X_train[split_indices[i]]
    y_trn, y_val = y_train[temp], y_train[split_indices[i]]
    
    model.fit(x_trn, y_trn, epochs = 7, batch_size =64)
    y_val_pred = model.predict(x_val)
    
    y_val_pred = lstm_utils.get_lable_from_ohe(y_val_pred)
    y_val_pred = lstm_utils.unroll_data(y_val_pred)
    
    y_val = lstm_utils.get_lable_from_ohe(y_val)
    y_val = lstm_utils.unroll_data(y_val)

    f1_score_cross_val_entity.append(f1_score(y_val, y_val_pred, average = 'macro'))
    score.append(model.evaluate(x_val, y_val))
    print("iteration done")

avg_score = np.mean(score, axis = 0)
std_score = np.std(score, axis =0)

model_json = model.to_json()
with open("model_ner.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_ner.h5")
print("Saved model to disk")

json_file = open('model_ner.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_ner.h5")
print("Loaded model from disk")  

intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("l3").output)
h_X = intermediate_layer_model.predict(X)

y_pred_ohe_test = model.predict(X_test)
y_pred_ohe = model.predict(X)

predicted_lables_test = lstm_utils.get_lable_from_ohe(y_pred_ohe_test)
true_lables_test = lstm_utils.get_lable_from_ohe(y_test)
predicted_lables = lstm_utils.get_lable_from_ohe(y_pred_ohe)
true_lables = lstm_utils.get_lable_from_ohe(y)

predicted_lables_test = lstm_utils.unroll_data(predicted_lables_test)
true_lables_test = lstm_utils.unroll_data(true_lables_test)
predicted_lables = lstm_utils.unroll_data(predicted_lables)
true_lables = lstm_utils.unroll_data(true_lables)

print("The classification report on test data : ")
print(classification_report(true_lables_test ,predicted_lables_test, labels = [0,1,2,3,4,5,6,7,8]))
print("The classification report on the entire dataset : ")
print(classification_report(true_lables ,predicted_lables, labels = [0,1,2,3,4,5,6,7,8]))

h_X_unrolled  = lstm_utils.unroll_hidden_features(h_X)

entity_indices, cleaned_entity_indices, word_list, word_list_sent = lstm_utils.get_lable_indices_by_sentences(corpus, 
                                                                                              predicted_lables)

d_a_k, d_k_a, d_k_b, d_b_k, h_a, h_b, rel_lables = lstm_utils.get_sdp_and_relation_lables(df_drug, df_effect, 
                                                                                          cleaned_entity_indices,
                                                                                          corpus, word_list, h_X_unrolled)

d_a_k_ohe_pca, d_k_a_ohe_pca, d_k_b_ohe_pca, d_b_k_ohe_pca = lstm_utils.get_sdp_encoding(d_a_k, d_k_a, d_k_b, d_b_k, 
                                                                                         apply_pca = True, 
                                                                                         n_component = 990)
        
data_ac, data_ca, data_cb, data_bc, rel_lables = lstm_utils.relation_lstm_data_generator(d_a_k_ohe_pca, d_k_a_ohe_pca, 
                                                                                         d_k_b_ohe_pca, d_b_k_ohe_pca,
                                                                                         h_a, h_b, rel_lables)

# =============================================================================
#      LSTM for Relationsship Classification
# =============================================================================

data_dim = 1590
timesteps = 1
num_classes = 1

x1 = Input(shape = (timesteps, data_dim))
l1 = Bidirectional(LSTM(300, return_sequences=True))(x1)

x2 = Input(shape = (timesteps, data_dim))
l2 = Bidirectional(LSTM(300, return_sequences=True))(x2)


x3 = Input(shape = (timesteps, data_dim))
l3 = Bidirectional(LSTM(300, return_sequences=True))(x3)

x4 = Input(shape = (timesteps, data_dim))
l4 = Bidirectional(LSTM(300, return_sequences=True))(x4)

output = concatenate([l1,l2,l3,l4],axis=-1) 

output = Dense(300, activation='tanh')(output)

output = Dense(1, activation='sigmoid')(output)

model_rel = Model(inputs = [x1,x2,x3,x4], outputs = output)
model_rel.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

train_split_per = 0.8

data_ac_train = data_ac[0:int(train_split_per*len(data_ac)),:,:]
data_ac_test = data_ac[int(train_split_per*len(data_ac)):,:,:]

data_ca_train = data_ca[0:int(train_split_per*len(data_ac)),:,:]
data_ca_test = data_ca[int(train_split_per*len(data_ac)):,:,:]

data_cb_train = data_cb[0:int(train_split_per*len(data_ac)),:,:]
data_cb_test = data_cb[int(train_split_per*len(data_ac)):,:,:]

data_bc_train = data_bc[0:int(train_split_per*len(data_ac)),:,:]
data_bc_test = data_bc[int(train_split_per*len(data_ac)):,:,:]

rel_lables_train = rel_lables[0:int(train_split_per*len(data_ac)),:,:]
rel_lables_test = rel_lables[int(train_split_per*len(data_ac)):,:,:]

model_rel.fit([data_ac_train, data_ca_train, data_cb_train, data_bc_train], rel_lables_train, batch_size=32, epochs=10)

split_indices_rel , lst_rel = lstm_utils.K_Fold(data_ac_train, n_splits = 3) 

score_rel = []
for i in range(len(split_indices_rel)):
    temp = copy.deepcopy(lst_rel)
    for value in split_indices_rel[i]:
        temp.remove(value)
    x_ac_trn, x_ac_val = data_ac_train[temp], data_ac_train[split_indices_rel[i]]
    x_ca_trn, x_ca_val = data_ca_train[temp], data_ca_train[split_indices_rel[i]]
    x_cb_trn, x_cb_val = data_cb_train[temp], data_cb_train[split_indices_rel[i]]
    x_bc_trn, x_bc_val = data_bc_train[temp], data_bc_train[split_indices_rel[i]]

    y_rel_trn, y_rel_val = rel_lables_train[temp], rel_lables_train[split_indices_rel[i]]
    
    model_rel.fit([x_ac_trn, x_ca_trn, x_cb_trn, x_bc_trn], y_rel_trn, epochs = 7, batch_size =64)
    score_rel.append(model_rel.evaluate([x_ac_val, x_ca_val, x_cb_val, x_bc_val], y_rel_val))
    print("iteration done")

avg_score_rel = np.mean(score_rel, axis = 0)
print("Average relation classification score = ", avg_score_rel[1])
std_score_rel = np.std(score_rel, axis =0)
print("Standard deviation for relation classification score = ", std_score_rel[1])

model_json = model_rel.to_json()
with open("model_rel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_rel.save_weights("model_rel.h5")
print("Saved model to disk")

json_file = open('model_rel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_rel = model_from_json(loaded_model_json)
# load weights into new model
model_rel.load_weights("model_rel.h5")
print("Loaded model from disk")  


predicted_lables_test = model_rel.predict([data_ac_test , data_ca_test , data_cb_test , data_bc_test ])

for i in range(len(predicted_lables_test)):
    if predicted_lables_test[i]>=0.5:
        predicted_lables_test[i] = 1
    else:
        predicted_lables_test[i]=0
        
print("F1 Score on test data is : ", f1_score(rel_lables_test[:,0,0], predicted_lables_test[:,0,0]))


plt.rcParams['figure.figsize'] = (10, 5)
plt.title("True Class Distribution")
plt.bar(list(class_distribution.keys()), [x / 943.40 for x in list(class_distribution.values())])
plt.rcParams['font.size'] = 7
plt.xlabel("Classes")
plt.ylabel("Percentage occurence of each class")
plt.show()

predicted_class_distribution = Counter(predicted_lables)
plt.title("Predicted Class Distribution")
plt.rcParams['figure.figsize'] = (10, 5)
plt.bar(list(class_distribution.keys()), [x / 943.40 for x in list(predicted_class_distribution.values())])
plt.xlabel("Classes")
plt.ylabel("Percentage occurence of each class")
plt.show()


