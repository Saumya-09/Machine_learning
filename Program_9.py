#Write a program to apply decision tree classifier on pima Indian Diabetes dataset.

#Code:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from google.colab import files
uploaded = files.upload()
data = pd.read_csv('diabetes.csv',header = 0)
data.head()
data.describe()
data.isnull().sum()
X_features = pd.DataFrame(data = data, columns = ["Glucose","BMI","Age"])
X_features.head(2)
#Considering the 3 features showing the max correlation. 
Y = data.iloc[:,8]
Y.head(3)
scaler = StandardScaler()
X_features = scaler.fit_transform(X_features)
X_features
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.25, random_state=10)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
models = []
models.append(("Decision Tree:",DecisionTreeClassifier()))
print('Model appended...')
results = []
names = []
for name,model in models:
    kfold = KFold(n_splits=5, random_state=3)
    cv_result = cross_val_score(model,X_train,Y_train, cv = kfold,scoring = "accuracy")
    names.append(name)
    results.append(cv_result)
for i in range(len(names)):
    print(names[i],results[i].mean()*100)
