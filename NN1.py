#%%
# load libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
from keras import backend as K
from sklearn.feature_extraction.text import CountVectorizer

#read dataset as dataframe
df = pd.read_table("Data/dialog_acts.dat",index_col=False,names=["words"])
label=[]
text=[]
#choose the column
line_list=df.iloc[:,0]
#create new dataframe to create new dataset for training 
dt=pd.DataFrame(columns=['dialogue','uttr'])
#extract the first word from string
for line in line_list:
    first_word= line.split()[0]
    uttr= line.replace("{} ".format(first_word),'')
    label.append(first_word.lower())
    text.append(uttr.lower())
  
#add dialogue and utterance to new dataframe
dt['dialogue']=label
dt['uttr']=text

#%%
# change output for labels NN
def change_label_NN(dt):
    label_NN = []
    for i in dt['dialogue']:
        if i == 'thankyou':
            label_NN.append(np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32))
        if i == 'ack':
            label_NN.append(np.asarray([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32))
        if i == 'affirm':
            label_NN.append(np.asarray([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32))
        if i == 'bye':
            label_NN.append(np.asarray([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32))
        if i == 'confirm':
            label_NN.append(np.asarray([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32))
        if i == 'deny':
            label_NN.append(np.asarray([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32))
        if i == 'hello':
            label_NN.append(np.asarray([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32))
        if i == 'inform':
            label_NN.append(np.asarray([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32))
        if i == 'negate':
            label_NN.append(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).astype(np.float32))
        if i == 'null':
            label_NN.append(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).astype(np.float32))
        if i == 'repeat':
            label_NN.append(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).astype(np.float32))
        if i == 'reqalts':
            label_NN.append(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).astype(np.float32))
        if i == 'reqmore':
            label_NN.append(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).astype(np.float32))
        if i == 'request':
            label_NN.append(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).astype(np.float32))
        if i == 'restart':
            label_NN.append(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).astype(np.float32))

    dt['NN_label'] = label_NN
    return dt

#%%
def vectorize(dt):
    dt = change_label_NN(dt)
    #vectorize features and labels to change the text to the number
    dt['dialogue_act_id'] = dt['dialogue'].factorize()[0]
    vectorizer = CountVectorizer(min_df=5, encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = vectorizer.fit_transform(dt['uttr']).toarray().astype(np.float32)
    labels = np.array(dt['NN_label'].tolist())
    #prepare dataset
    x_train, x_test, y_train, y_test = train_test_split(features,labels, test_size=0.15, random_state=0)

    tf.convert_to_tensor(x_train)
    tf.convert_to_tensor(x_test)
    tf.convert_to_tensor(y_train)
    tf.convert_to_tensor(y_test)

    return x_train, x_test, y_train, y_test

#%%
#add metrics
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#create model
def create_model(dt):
    x_train, x_test, y_train, y_test = vectorize(dt)
    input_dim = x_train.shape[1]  # Number of features
    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(15, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1_m,precision_m, recall_m])
    # model.summary()
    # train model
    model.fit(x_train, y_train, epochs=10, verbose=True, validation_data=(x_test, y_test), batch_size=10)
    # loss, accuracy, f1_score, precision, recall = model.evaluate(x_train, y_train, verbose=True)
    # print("Accuracy: {:.4f}".format(accuracy))
    return model



class NeuralNet:
    def __init__(self, dt):
        model = create_model(dt)
        return model

    def predict(model, sentence):
        prediction = model.predict(sentence)
        return prediction

model = create_model(dt)
model.summary()

#%%
def predict(model, sentence):
    prediction = model.predict(sentence)
    return prediction
predict(model, 'I thank you')
# %%
