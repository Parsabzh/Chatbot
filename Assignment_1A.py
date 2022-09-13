
# load libraries 
import numpy as np
import pandas as pd
import sklearn 

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
print(dt)

