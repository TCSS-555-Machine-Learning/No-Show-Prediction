# Load libraries

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.parser import parse
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

cwd = os.getcwd()
print('Current folder is {}'.format(cwd))


def get_gender_classification(gender):
    if gender == 'F':
        return 0
    elif gender == 'M':
        return 1
    else:
        return -1


def get_noshow_classification(noshow):
    if noshow == 'No':
        return 0
    elif noshow == 'Yes':
        return 1
    else:
        return -1


def get_neighbourhood_classification(neighbour):
    i, = np.where(neighbourhood == neighbour)
    return i[0]


def get_days(sch, app):
    a = app[0:10]
    b = parse(a)
    c = sch[0:10]
    d = parse(c)
    return b - d


def get_age_classification(age):
    if age <= 1:
        return age
    elif age < 65:
        return 2
    else:
        return 3


def add_data(row):
    row.gender = get_gender_classification(row.gender)
    row.noshow = get_noshow_classification(row.noshow)
    row.neighbourhood = get_neighbourhood_classification(row.neighbourhood)
    row.agegroup = get_age_classification(row.age)
    day1 = parse(row.scheduledday[0:10])
    day2 = parse(row.appointmentday[0:10])
    row.daysbefore = (day2 - day1).days
    row.appointmentdayofweek = datetime.weekday(day2)
    return row


# Load dataset
data = pd.read_csv("Data\KaggleV2-May-2016.csv")
data = shuffle(data)
data = data[:5000]  # get first 5000 rows

# Dimensions of dataset
n = data.shape[0]  # rows
p = data.shape[1]  # columns

# Lower case all column header
data.columns = [x.lower() for x in data.columns]

# Rename several DataFrame columns
data = data.rename(columns={
    'sms_received': 'smsreceived',
    'no-show': 'noshow',
})

print('n = {} and p = {}'.format(n, p))

# shape
print('Dataset''s shape: {}'.format(data.shape))

# add 2 columns
data.insert(7, 'agegroup', 0)
data.insert(5, 'daysbefore', 0)  # how many days from scheduled day to appointment day
data.insert(5, 'appointmentdayofweek', 0)  # what is the day of week of the appointment day

# head
print(data.head(2))

# descriptions
# print(data.describe())


neighbourhood = data.neighbourhood.unique()
print(neighbourhood)

# this step appplies the add_data function on every row.
data = data.apply(lambda row: add_data(row), axis=1)

print(data.head(5))
data.groupby(['appointmentdayofweek', 'noshow']).size().reset_index(name='counts')

# drop columns that we don't need
# data.drop('patientid', axis=1, inplace=True)
data.drop('appointmentid', axis=1, inplace=True)  # axis=1 means apply for each row
data.drop('appointmentday', axis=1, inplace=True)  # axis=1 means apply for each row
data.drop('scheduledday', axis=1, inplace=True)  # axis=1 means apply for each row
data.drop('appointmentdayofweek', axis=1, inplace=True)  # axis=1 means apply for each row
data.drop('age', axis=1, inplace=True)  # axis=1 means apply for each row
# data.drop('agegroup', axis=1, inplace=True) #axis=1 means apply for each row 
# data.drop('daysbefore', axis=1, inplace=True) #axis=1 means apply for each row 
data.drop('neighbourhood', axis=1, inplace=True)  # axis=1 means apply for each row
data.drop('alcoholism', axis=1, inplace=True)  # axis=1 means apply for each row
data.drop('diabetes', axis=1, inplace=True)  # axis=1 means apply for each row
# data.drop('gender', axis=1, inplace=True) #axis=1 means apply for each row
data.drop('handcap', axis=1, inplace=True)  # axis=1 means apply for each row
data.drop('scholarship', axis=1, inplace=True)  # axis=1 means apply for each row
data.drop('hipertension', axis=1, inplace=True)  # axis=1 means apply for each row
data.drop('smsreceived', axis=1, inplace=True)  # axis=1 means apply for each row

# checking
print(data.head(2))
print('n = {} and p = {}'.format(data.shape[0], data.shape[1]))

# Split-out validation dataset (No-show column only)
col = len(data.columns) - 1
array = data.values  # numpy array
X = array[:, 0:col]  # numpy array - all columns except the last column (noshow)
Y = array[:, col]  # numpy array - the last noshow column
print(X[0:2, ])  # print top 2 rows
print(Y[0:2, ])  # print top 2 rows
print(X.size)

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

print('{}'.format(X_train, Y_train))

# get train set and validation set
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

print('{}'.format(X_train, Y_train))

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
