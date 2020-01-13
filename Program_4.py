#Write a program to predict total payment for given number of claims on Swedish auto insurance dataset using linear regression.
#Code:

from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

def best_fit_slope(X,y):
    slope_m = ((mean(X)*mean(y)) - mean(X*y))/(mean(X)**2 - mean(X**2))
    bias_b = mean(y) - slope_m*mean(X)
    return slope_m, bias_b

from google.colab import files
uploaded = files.upload()



df = pd.read_excel('slr06.xls')
df.head()
df.describe()

X = np.array(df['X'], dtype=np.float64)
y = np.array(df['Y'], dtype=np.float64)

fig,ax = plt.subplots()
ax.scatter(X,y)
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('Input Data Scatter Plot')

 m,b = best_fit_slope(X,y)

print('Slope: ',m)
print('Bias: ',b)

y_hat = m*X + b
print('y_hat: ', y_hat)

fig,ax = plt.subplots()
ax.scatter(X,y)
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.plot(X,y_hat)
ax.set_title('Line fit to Input Data')
