import pandas as pd
import librosa
import numpy as np
import csv

""" Training Model """
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

f = open('data.csv')
csv_f = csv.reader(f)
# train=pd.read_csv('data.csv')
# train.head()

def parser(row):
    for row in csv_f:
        filename=str(row[0])+'.wav'
        print(filename)
        x,s=librosa.load(filename,res_type='kaiser_fast')
        mfccs=np.mean(librosa.feature.mfcc(y=x,sr=s,n_mfcc=40).T,axis=0)
        feature=mfccs
        label=row[1]
        return [feature, label]

temp=train.apply(parser,axis=1)
print(type(temp))
temp.columns=['feature','label']

print(temps)

X=np.array(temp.feature.tolist())
Y=np.array(temp.label.tolist())
lb=LabelEncoder()
yy=np_utils.to_categorical(lb.fit_transform(Y))

num_labels = yy.shape[1]
filter_size = 2

""" Build Model """
model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


""" Training Model """
aTrain,aTest,bTrain,bTest=train_test_split(X,yy,test_size=0.2)
model.fit(aTrain, bTrain, batch_size=32, epochs=100, validation_data=(aTest, bTest))