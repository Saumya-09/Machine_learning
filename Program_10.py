#Design and implement a neural network with Pima Indian diabetes dataset.

#Code:

from keras.models import Sequential
from keras.layers import Dense
from numpy import loadtxt
from sklearn.model_selection import train_test_split

dataset = loadtxt('pima-indians-diabetes.csv',delimiter=',')
X = dataset[:,0:8]
Y = dataset[:,8]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.7)

dataset.shape

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=231,batch_size=10,validation_split=0.2)

_, accuracy = model.evaluate(X_test,Y_test)
print('Accuracy: %2f'%(accuracy*100))
