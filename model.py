import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import os

import numpy as np

########Initialise random forest

local_path = (os.path.dirname(os.path.realpath('__file__')))

file_name = ('data.csv')  # file of total data
data_path = os.path.join(local_path, file_name)
print(data_path)
df = pd.read_csv(r'' + data_path)

print(df)

units_in_data = 19  # no. of units in data

titles = []
for i in range(units_in_data):
    titles.append("unit-" + str(i))
X = df[titles]
y = df['letter']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

clf = RandomForestClassifier(n_estimators=30)  # random forest
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
cmrf = confusion_matrix(y_test, y_pred)
print("1.Random Forest Accuracy")

print("Random Forest classification_report")
print(classification_report(y_pred, y_test, labels=None))
print("Random Forest confusion_matrix")
print(cmrf)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("CONFUSION MATRIX OF RF")
print(cm)
tpr = float(cm[0][0]) / np.sum(cm[0])
fpr = float(cm[1][1]) / np.sum(cm[1])
plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Confusion Matrix of RF '
plt.title(all_sample_title, size=15);
plt.show()


#Neural network module
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
#from tensorflow.keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

#Change the label to one hot vector
'''
[0]--->[1 0 0]
[1]--->[0 1 0]
[2]--->[0 0 1]
'''
y_train=np_utils.to_categorical(y_train,num_classes=2)
y_test=np_utils.to_categorical(y_test,num_classes=2)
print("Shape of y_train",y_train.shape)
print("Shape of y_test",y_test.shape)

# We are using keras library for neural network. Input_dim =15 means we have 15 independent features.
model=Sequential()
model.add(Dense(1000,input_dim=15,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(6,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())

ann=model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=20,epochs=58,verbose=1)
# save the model to disk
