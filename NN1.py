#%%
# load libraries 
from tkinter.font import _MetricsDict
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras import layers
from keras.backend import clear_session
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

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

#%%
#vectorize features and labels to change the text to the number
dt['dialogue_act_id'] = dt['dialogue'].factorize()[0]
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(dt['uttr']).toarray().astype(np.float32)
labels = np.array(dt['NN_label'].tolist())
#prepare dataset
x_train, x_test, y_train, y_test = train_test_split(features,labels, test_size=0.15, random_state=0)

tf.convert_to_tensor(x_train)
tf.convert_to_tensor(x_test)
tf.convert_to_tensor(y_train)
tf.convert_to_tensor(y_test)

#%%
#create model
input_dim = x_train.shape[1]  # Number of features
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(15, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#%%
# train model
history = model.fit(x_train, y_train, epochs=10, verbose=True, validation_data=(x_test, y_test), batch_size=10)
loss, accuracy = model.evaluate(x_train, y_train, verbose=True)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=True)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# %%
# plot results
print(history.history.keys())
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

plot_history(history)
# %%
# plot confusion matrix
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
y_pred = model.predict(x_test)
matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=None)
disp.plot()
# %%
