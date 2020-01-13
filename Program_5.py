#Write a python program to perform multiclass classification on iris dataset.
#Code:

from sklearn import datasets
iris=datasets.load_iris()
x=iris.data
y=iris.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5)
from sklearn import tree
classifier=tree.DecisionTreeClassifier()
classifier.fit(x_train,y_train)
predictions=classifier.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))


classifier = classifier.fit(iris.data, iris.target)
tree.export_graphviz(classifier) 


dot_data = tree.export_graphviz(classifier, out_file=None, 
feature_names=iris.feature_names, 
class_names=iris.target_names)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data) 

# Show graph
Image(graph.create_png())
