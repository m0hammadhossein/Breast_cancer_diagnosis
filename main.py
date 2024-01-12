import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from calc_metrics import calc_metrics

# Set library matplotlib
matplotlib.use('TKAgg')

# Load dataset
bc = load_breast_cancer()

# Preprocessing
x_train, x_test, y_train, y_test = train_test_split(bc.data, bc.target, test_size=0.2)
scaler = MinMaxScaler(feature_range=(0, 1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Classification Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pre_train = gnb.predict(x_train)
y_pre_test = gnb.predict(x_test)
acc_train_gnb, acc_test_gnb, pr_gnb, rc_gnb = calc_metrics(y_train, y_test, y_pre_train, y_pre_test)

# Classification KNN
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train, y_train)
y_pre_train = knn.predict(x_train)
y_pre_test = knn.predict(x_test)
acc_train_knn, acc_test_knn, pr_knn, rc_knn = calc_metrics(y_train, y_test, y_pre_train, y_pre_test)

# Classification DecisionTree

dt = DecisionTreeClassifier(max_depth=60)
dt.fit(x_train, y_train)
y_pre_train = dt.predict(x_train)
y_pre_test = dt.predict(x_test)
acc_train_dt, acc_test_dt, pr_dt, rc_dt = calc_metrics(y_train, y_test, y_pre_train, y_pre_test)

# Classification ًRandom Forest

rf = RandomForestClassifier(n_estimators=50)
rf.fit(x_train, y_train)
y_pre_train = rf.predict(x_train)
y_pre_test = rf.predict(x_test)
acc_train_rf, acc_test_rf, pr_rf, rc_rf = calc_metrics(y_train, y_test, y_pre_train, y_pre_test)

# Classification ًSVM

svm = SVC(kernel='poly')
svm.fit(x_train, y_train)
y_pre_train = svm.predict(x_train)
y_pre_test = svm.predict(x_test)
acc_train_svm, acc_test_svm, pr_svm, rc_svm = calc_metrics(y_train, y_test, y_pre_train, y_pre_test)

# Classification ًLogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pre_train = lr.predict(x_train)
y_pre_test = lr.predict(x_test)
acc_train_lr, acc_test_lr, pr_lr, rc_lr = calc_metrics(y_train, y_test, y_pre_train, y_pre_test)

# Classification ًANN

ann = MLPClassifier(hidden_layer_sizes=(20, 20), activation='relu', max_iter=64)
ann.fit(x_train, y_train)
y_pre_train = ann.predict(x_train)
y_pre_test = ann.predict(x_test)
acc_train_ann, acc_test_ann, pr_ann, rc_ann = calc_metrics(y_train, y_test, y_pre_train, y_pre_test)

# Comparison

acc_train = [acc_train_gnb, acc_train_knn, acc_train_dt, acc_train_rf, acc_train_svm, acc_train_lr, acc_train_ann]
acc_test = [acc_test_gnb, acc_test_knn, acc_test_dt, acc_test_rf, acc_test_svm, acc_test_lr, acc_test_ann]
pr_test = [pr_gnb, pr_knn, pr_dt, pr_rf, pr_svm, pr_lr, pr_ann]
rc_test = [rc_gnb, rc_knn, rc_dt, rc_rf, rc_svm, rc_lr, rc_ann]
title = ['GNB', 'KNN', 'DT', 'RF', 'SVM', 'LR', 'ANN']
colors = ['black', 'red', 'yellow', 'orange', 'green', 'blue', 'pink']
plt.figure(figsize=(6, 6))
plt.subplot(221)
plt.xticks(fontsize=7)
plt.title('Accuracy train', size=7)
plt.bar(title, acc_train, color=colors)
plt.subplot(222)
plt.xticks(fontsize=7)
plt.title('Accuracy test', size=7)
plt.bar(title, acc_test, color=colors)
plt.subplot(223)
plt.xticks(fontsize=7)
plt.title('Precision test', size=7)
plt.bar(title, pr_test, color=colors)
plt.subplot(224)
plt.xticks(fontsize=7)
plt.title('Recall test', size=7)
plt.bar(title, rc_test, color=colors)
plt.show()
