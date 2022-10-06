# load libraries
from typing_extensions import Self
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
from keras import backend as K
from keras.layers import Dropout
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


def create_dataframe():
    # read dataset as dataframe
    df = pd.read_table("Data/dialog_acts.dat", index_col=False, names=["words"])
    label = []
    text = []
    # choose the column
    line_list = df.iloc[:, 0]
    # create new dataframe to create new dataset for training
    dt = pd.DataFrame(columns=['dialogue', 'uttr'])
    # extract the first word from string
    for line in line_list:
        first_word = line.split()[0]
        uttr = line.replace("{} ".format(first_word), '')
        label.append(first_word.lower())
        text.append(uttr.lower())

    # add dialogue and utterance to new dataframe
    dt['dialogue'] = label
    dt['uttr'] = text
    return dt

dt = create_dataframe()

# change output for labels NN
def change_label_NN(dt):
    label_NN = []
    classifiers = ['thankyou', 'ack', 'affirm', 'bye', 'confirm', 'deny',
     'hello', 'inform', 'negate', 'null', 'repeat', 'reqalts', 'request', 'restart']

    for i in dt['dialogue']:
        array = np.zeros(15)
        for j in range(len(classifiers)):
            if i == classifiers[j]:
                np.put(array, j, 1)
        label_NN.append(np.asarray(array).astype(np.float32))

    dt['NN_label'] = label_NN
    return dt

def vectorize(dt):
    dt = change_label_NN(dt)
    #vectorize features and labels to change the text to the number
    dt['dialogue_act_id'] = dt['dialogue'].factorize()[0]
    vectorizer = CountVectorizer(min_df=5, encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = vectorizer.fit_transform(dt['uttr']).toarray().astype(np.float32)
    
    pickle.dump(vectorizer, open("vector.pickel", "wb"))
    
    labels = np.array(dt['NN_label'].tolist())
    #prepare dataset
    x_train, x_test, y_train, y_test = train_test_split(features,labels, test_size=0.15, random_state=0)

    tf.convert_to_tensor(x_train)
    tf.convert_to_tensor(x_test)
    tf.convert_to_tensor(y_train)
    tf.convert_to_tensor(y_test)

    return x_train, x_test, y_train, y_test

#add metrics
def recall_m(y_true, y_pred):
    # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    # recall = true_positives / (possible_positives + K.epsilon())
    recall= recall_score(y_true,y_pred)
    return recall

def precision_m(y_true, y_pred):
    # true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # precision = true_positives / (predicted_positives + K.epsilon())
    precision=precision_score(y_true,y_pred,)
    return precision

def f1_m(y_true, y_pred):
    # precision = precision_m(y_true, y_pred)
    # recall = recall_m(y_true, y_pred)
    # return 2*((precision*recall)/(precision+recall+K.epsilon()))
    f1=f1_score(y_true,y_pred)
    return f1

#create model
def create_model(dt):
    x_train, x_test, y_train, y_test = vectorize(dt)
    input_dim = x_train.shape[1]  # Number of features
    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2, input_shape=(input_dim,)))
    model.add(layers.Dense(128, input_dim=input_dim, activation='relu'))
    # model.add(Dropout(0.2, input_shape=(input_dim,)))
    model.add(layers.Dense(64, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(15, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history= model.fit(x_train, y_train, epochs=100, verbose=True, validation_data=(x_test, y_test), batch_size=10)
    #save model
    model.save('NeuralNet.h5')
    print(history.history.keys())
    #print accuracy graph 
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    #plot loss graph
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    return model


class NeuralNet:
    def __init__(self):
        # self.load_model()
        self.model = self.load_model()
        return

    def create_NN_model(dt):
        create_model(dt)
        return
    
    def load_model(self):
        # self.model = tf.keras.models.load_model('NeuralNet.h5')
        return tf.keras.models.load_model('NeuralNet.h5')

    def predict(self, sentence):
        sentence = [sentence]
        vectorizer = pickle.load(open("vector.pickel", "rb"))
        vector = vectorizer.transform(sentence)
        prediction = self.model.predict(vector, verbose=0)
        classifiers = ['thankyou', 'ack', 'affirm', 'bye', 'confirm', 'deny',
         'hello', 'inform', 'negate', 'null', 'repeat', 'reqalts', 'request', 'restart']
        index = np.argmax(prediction[0], axis=0)
        return classifiers[index]
