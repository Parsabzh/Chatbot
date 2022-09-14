
# load libraries 
import numpy as np
import pandas as pd
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

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

#prepare trainset and testset 
x_train, x_test, y_train, y_test = train_test_split(dt['dialogue'], dt['uttr'], test_size=0.15, random_state=0)
#define model
lr = LogisticRegression()
#fit the model
lr.fit(x_train, y_train)
#make a prediction
predictions = lr.predict(x_test)
score = lr.score(x_test, y_test)
print(score)

