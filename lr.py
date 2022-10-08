# load libraries 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#read dataset as dataframe
def create_dataset():
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
    return dt
def vectorize(dt):
    #vectorize feactures and labels to change the text to the number
    dt['dialogue_act_id'] = dt['dialogue'].factorize()[0]
    vectorizer = CountVectorizer(min_df=5, encoding='latin-1', ngram_range=(1, 5), stop_words='english')
    features = pd.DataFrame(vectorizer.fit_transform(dt['uttr'].values).toarray().astype(np.float32)).values
    labels = dt['dialogue_act_id'].values
     #prepare dataset
    x_train, x_test, y_train, y_test = train_test_split(features,labels, test_size=0.15, random_state=0)
    return x_train, x_test, y_train, y_test 
def train(x_train, x_test, y_train, y_test):
   
    # define model and fit it
    lr = LogisticRegression(penalty='l2',solver='saga')
    lr.fit(x_train, y_train)
    filename = 'lr_model.sav'
    pickle.dump(lr, open(filename, 'wb'))
    score = lr.score(x_test, y_test)
    print(score)
def predict_lr(x_test):
    model = pickle.load(open('lr_model.sav', 'rb'))
    y_pred= model.predict(x_test)
    return y_pred
dt = create_dataset()


# print the accuracy, precision, recall, and f1 score for the predicted dialogue classes and plot a confusion matrix
def calculate_metrics(y_pred, y_test):
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation='vertical')
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("Accuracy: " + str(acc) + " ,F1: " + str(f1), " ,Recall: " + str(recall), " ,Precision: " + str(precision))
    plt.show()
    print(disp)

x_train, x_test, y_train, y_test = vectorize(dt)
y_pred=predict_lr(x_test)
calculate_metrics(y_pred,y_test)