#Implement multiclass classification with neural network on Iris flower species.

#Code:

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

dataset = load_iris()
X = dataset.data
Y = dataset.target

LE = LabelEncoder()
LE.fit(Y)
e_Y = LE.transform(Y)
Y = np_utils.to_categorical(e_Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8)

model = Sequential()
model.add(Dense(7,input_dim=4,activation='sigmoid'))
model.add(Dense(3,activation='softmax'))

model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

import matplotlib.pyplot as plt
history = model.fit(X_train,Y_train,epochs=20,batch_size=5,validation_split=0.2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

_, accuracy = model.evaluate(X_test,Y_test)
