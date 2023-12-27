
import keras
import nltk
import pickle
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
import random
from googlesearch import *
nltk.download('punkt')
nltk.download('wordnet')
from keras.callbacks import Callback
from nltk.stem import WordNetLemmatizer
import streamlit as st
lemmatizer=WordNetLemmatizer()
words=[]
classes=[]
documents=[]
ignore=['?','!',',',"'s"]

with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)
# Đọc tệp intents.json với encoding UTF-8

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w,intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
words=[lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore]
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern = doc[0]
    pattern = [lemmatizer.lemmatize(word.lower()) for word in pattern]

    for word in words:
        if word in pattern:
            bag.append(1)
        else:
            bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
    
random.shuffle(training)
#training = np.array(training)
training = np.array([(np.array(bag), np.array(output_row)) for bag, output_row in training], dtype=object)



#X_train=list(training[:,0])
#y_train=list(training[:,1])  
X_train = training[:, 0].tolist()
y_train = training[:, 1].tolist()

#Model
"""
model=Sequential()
model.add(Dense(128,activation='relu',input_shape=(len(X_train[0]),)))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]),activation='softmax'))
"""
#mới
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(len(X_train[0]),)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

class TrainingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs['accuracy'] * 100
        loss = logs['loss']
        print("\n"f'Epoch {epoch+1} - loss: {loss:.4f} - accuracy: {accuracy:.2f}%')
        
callback = TrainingCallback()
adam=keras.optimizers.Adam(0.001)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

#model.fit(np.array(X_train),np.array(y_train),epochs=200,batch_size=10,verbose=1)
model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=10, verbose=1, callbacks=[callback])
model.save('mymodel.h5')
