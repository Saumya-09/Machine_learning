#Write a python program to classify various types of from with iris dataset using support vector machine.

#Code:

from sklearn import datasets
iris = datasets.load_iris()
print(&quot;Features: &quot;, iris.feature_names)
print(&quot;Labels: &quot;, iris.target_names)
iris.data.shape
print(iris.target)
from sklearn import svm
import numpy as np
from scipy import stats
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split

%matplotlib inline
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
test_size=0.3,random_state=109)

clf = svm.SVC(kernel=&#39;linear&#39;)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
print(&quot;Accuracy:&quot;,metrics.accuracy_score(y_test, y_pred))

from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=50, centers=2,
random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=&#39;autumn&#39;);
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=&#39;autumn&#39;)
plt.plot([0.6], [2.1], &#39;x&#39;, color=&#39;red&#39;, markeredgewidth=2, markersize=10)

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
plt.plot(xfit, m * xfit + b, &#39;-k&#39;)
plt.xlim(-1, 3.5);
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=&#39;autumn&#39;)
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
yfit = m * xfit + b
plt.plot(xfit, yfit, &#39;-k&#39;)
plt.fill_between(xfit, yfit - d, yfit + d, edgecolor=&#39;none&#39;,
color=&#39;#AAAAAA&#39;, alpha=0.4)
plt.xlim(-1, 3.5);
