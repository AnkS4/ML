from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score as acc
import numpy
import pandas

#Classifiers
dt = tree.DecisionTreeClassifier()
knn = KNeighborsClassifier()
svc_linear = SVC(kernel='linear')
svc_poly = SVC(kernel='poly')
svc_rbf = SVC(kernel='rbf')
nb = GaussianNB()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#test
Z = [[190, 70, 43]]

#fitting data
dt = dt.fit(X, Y)
knn = knn.fit(X, Y)
svc_linear = svc_linear.fit(X, Y)
svc_poly = svc_poly.fit(X, Y)
svc_rbf = svc_rbf.fit(X, Y)
nb = nb.fit(X, Y)

#Predicting with same data (as smaller dataset availability)
pred_dt = dt.predict(X)
pred_knn = knn.predict(X)
pred_svc_linear = svc_linear.predict(X)
pred_svc_poly = svc_poly.predict(X)
pred_svc_rbf = svc_rbf.predict(X)
pred_nb = nb.predict(X)

#Accuracy
acc_dt = acc(Y, pred_dt)
acc_knn = acc(Y, pred_knn)
acc_svc_linear = acc(Y, pred_svc_linear)
acc_svc_poly = acc(Y, pred_svc_poly)
acc_svc_rbf = acc(Y, pred_svc_rbf)
acc_nb = acc(Y, pred_nb)

gender = pandas.DataFrame({'name': ['Decision Tree', 'KNN', 'SVC Linear', 'SVC Polynomial', 'SVC RBF', 'Gaussian Naive Bayes'],\
'accuracy': [acc_dt, acc_knn, acc_svc_linear, acc_svc_poly, acc_svc_rbf, acc_nb],\
'prediction': [dt.predict(Z), knn.predict(Z), svc_linear.predict(Z), svc_poly.predict(Z), svc_rbf.predict(Z), nb.predict(Z)]})

#Accuracy scores non-ascending with prediction
print((gender.sort_values(by='accuracy', ascending=0)).to_string(index=False))