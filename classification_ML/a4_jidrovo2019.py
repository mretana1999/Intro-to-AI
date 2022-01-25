import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from sklearn import tree
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

# ################
# Code used as a reference and starting point: Code from the Prof. Oge Marques
# URL:
# https://colab.research.google.com/drive/1Mz6ZUi2VXCsQCCZ0GiwS3YNUthcrPslz?usp=sharing&authuser=1
# ################

# our first step here is to load the necessary data bu using the read_csv command from pandas library to load the
# Iris dataset
cols = ['sepal_length', ' sepal_width', 'petal_length', 'petal_width', 'class']
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', names=cols)

# after the data is loaded we will show the top of the dataset
print("This is the top of the dataset we have loaded: ")
print(iris.head())

# After loading the dataset into a dataframe in memory, the next step is to perform an exploratory data analysis. The
# objective of the EDA is to discover as much information as possible about the dataset.

# We will begin by using the describe() method, the describe() is a good starting point because it prints statistics
# of the dataset, like mean, standard deviation, etc.

print('\nThese are statistics of the dataset. This includes information such as mean, SD, and min: ')
print(iris.describe())

# The next step in our process will be to write some code that will allow us to generate a pair plot for the Iris
# dataset. The seaborn library has an amazingly useful function for this exact purpose and it is what we will be
# utilizing today. We will be using the pairplot() function which will allow us to generate an insightful pairplot.

# here we generate the pairplot
sns.pairplot(iris, hue='class')

# and here we will use the matplotlib command show() to display the plot
# plt.show()

# we found it optimal to color the plot on the basis of class to give more insight into the information we are
# seeing. so on our plot there are different colors / hues for the three different classes of iris (iris versicolor,
# iris setosa, and iris virginica)

# the next step in the process will be to preprocess the data this is an important step because while we like to
# think that our data is 'perfect' is usually not and we need to ensure that it is ready to be consumed by a machine
# learning algorithm. So in this step we will be doing some prep work on our data before we go ahead and use it.
# Our textbook, Grokking Artificial Intelligence Algorithms, referred to this step as Clean and Wrangle, stating that
# 'real-world data is never ideal to work with'.

# In the context of our algorithm, we will be converting the string values that exist for our class column to integer
# numbers because the algorithm we intend to use does not process string values.

iris['class'].unique()

iris['class_encod'] = iris['class'].apply(lambda x: 0 if x == 'Iris-setosa' else 1 if x == 'Iris-versicolor' else 2)
iris['class_encod'].unique()

y = iris[['class_encod']]  # target attributes
X = iris.iloc[:, 0:4]  # input attributes

print('\nThese are our target attributes: ')
print(X.head())
print('\nThese are our input attributes: ')
print(y.head())

# the next step in our process will be to normalize the features of the Iris dataset so that all attributes fit
# within the [0..1] range.

scaler = preprocessing.MinMaxScaler()
trfm = scaler.fit_transform(X)
newScaled = pd.DataFrame(trfm, columns=X.columns)
X = newScaled

print('\nHere is our features of the Iris dataset scaled so that all attributes fit within the [0..1] range: ')
print(X.describe())

# Now we are on to the 'meat and potatoes' where we will select an algorithm and train the model

# we will begin by setting our seed for reproducibility
random.seed(42)

# here we will use the method train_test_split() to split the X and y dataframes in training data and test data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0, stratify=y)
print('\nOutput after splitting x and y dataframes into training data and testing data:')
print(np.shape(y_train))

# Now we will use X_train and y_train to build a KNN (K-nearest-neighbors) classifier, using the KNeighborsClassifier
# class provided by scikit-learn.

m = KNeighborsClassifier()
m.fit(X_train, np.ravel(y_train))

# With the model built, we can make use of the predict() method to calculate the predicted category of an instance
# We will predict the class of the first 10 lines of the x_test dataset

print('\nPrediction of the class of the first 10 lines of the x_test dataset - KNN: ')
print(m.predict(X_test.iloc[0:10]))

print(y_test[0:10])

# Using methods like `score()` and `confusion_matrix()`, we can measure the performance of our model. We see that the
# accuracy of our model is very close to 100%, which means that the model predicted correctly almost all cases of the
# test dataset.

print('\nThe accuracy of our KNN model: ')
print(m.score(X_test, y_test))

# Now we will generate a confusion matrix to show where errors occurred (which classes were misclassified)
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    dispKNN = plot_confusion_matrix(m, X_test, y_test,
                                    display_labels=iris['class'].unique(),
                                    cmap=plt.cm.Blues,
                                    normalize=normalize)
    dispKNN.ax_.set_title(title)

# plt.show()

# We might need our model later so we will use the dump() method to save the model to a file
dump(m, 'iris-classifier-knn.dmp')

# to load the model again we can use the following code:

ic_knn = load('iris-classifier-knn.dmp')
print('\nKNN confusion matrix results: ')
print(confusion_matrix(y_test, ic_knn.predict(X_test)))

# print(confusion_matrix(y_test, ic.predict(X_test)))

# The next step in our process will be to build a decision tree classifier, using the DecisionTreeClassifier class
# provided by scikit-learn.

# Just as before, our code should build, train, and test the classifier, compute its accuracy, display the confusion
# matrices, save the model to a file for later use, load it from file and confirm that it's working.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0, stratify=y)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

print('\nPrediction of the class of the first 10 lines of the x_test dataset - DTC: ')
print(clf.predict(X_test.iloc[0:10]))

print(y_test[0:10])

print('\nThe accuracy of our DTC model: ')
print(clf.score(X_test, y_test))

# Here we will set up the non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    dispDTC = plot_confusion_matrix(clf, X_test, y_test,
                                    display_labels=iris['class'].unique(),
                                    cmap=plt.cm.Blues,
                                    normalize=normalize)
    dispDTC.ax_.set_title(title)

# and then show it
# plt.show()


# and save the model for later use
dump(clf, 'iris-classifier-dtc.dmp')

# we can load this model just like before with:
ic_dtc = load('iris-classifier-dtc.dmp')

print('\nDTC confusion matrix results: ')
print(confusion_matrix(y_test, ic_dtc.predict(X_test)))

# we can now plot the tree
plt.figure(figsize=(12, 12))
tree.plot_tree(clf, feature_names=X.columns, class_names=iris['class'].unique(), filled=True, fontsize=8)

# plt.show()

testText = """
##############################
#                            #
#          Testing           #
#                            #
##############################
"""

print(testText)

# Per our class textbook, an 80/20 split for training and testing data is usually done. "Training and testing data
# are usually split 80/20, with 80% of the available data used as training data and 20% used to test the model "


# In prior cases, however, we used a 70/30 split for our training and testing models. This got my group interested in
# seeing what the effects of different extremes of these splits may be. Also, while we know what the result of this
# endeavor would look like, we also wanted to see what the result of using the same data to train and test would
# look like

# First We will start with the KNN Classifier

print('##############BaseLine####################')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0, stratify=y)
m = KNeighborsClassifier()
m.fit(X_train, np.ravel(y_train))

print('\nThe accuracy of our KNN base model: ')
print(m.score(X_test, y_test))

print('\nKNN - Run 1: 80/20 split')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0, stratify=y)
m = KNeighborsClassifier()
m.fit(X_train, np.ravel(y_train))

print('\nThe accuracy of our KNN model - (80/20): ')
print(m.score(X_test, y_test))

print('\nKNN - Run 2: 90/10 split')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=0, stratify=y)
m = KNeighborsClassifier()
m.fit(X_train, np.ravel(y_train))

print('\nThe accuracy of our KNN model - (90/10): ')
print(m.score(X_test, y_test))

print('\nKNN - Run 3: 50/50 split')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                    random_state=0, stratify=y)
m = KNeighborsClassifier()
m.fit(X_train, np.ravel(y_train))

print('\nThe accuracy of our KNN model - (50/50): ')
print(m.score(X_test, y_test))

print('\nKNN - Run 4: 10/90 split')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9,
                                                    random_state=0, stratify=y)
m = KNeighborsClassifier()
m.fit(X_train, np.ravel(y_train))

print('\nThe accuracy of our KNN model - (10/90): ')
print(m.score(X_test, y_test))

# Now we will move on to the decision tree classifier


print("\nDecision Tree Classifier")
print('##############BaseLine####################')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0, stratify=y)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

print('\nThe accuracy of our DTC base model: ')
print(clf.score(X_test, y_test))

print('\nDTC - Run 1: 80/20 split')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0, stratify=y)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

print('\nThe accuracy of our DTC model - (80/20): ')
print(clf.score(X_test, y_test))

print('\nDTC - Run 2: 50/50 split')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                    random_state=0, stratify=y)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

print('\nThe accuracy of our DTC model - (50/50): ')
print(clf.score(X_test, y_test))

print('\nDTC - Run 3: 90/10 split')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=0, stratify=y)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

print('\nThe accuracy of our DTC model - (90/10): ')
print(clf.score(X_test, y_test))

print('\nDTC - Run 3: 10/90 split')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9,
                                                    random_state=0, stratify=y)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

print('\nThe accuracy of our DTC model - (10/90): ')
print(clf.score(X_test, y_test))

