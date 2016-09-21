# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 09:24:49 2016

@author: Jae

Analyzed the data set on student's performance and  develop a model that will predict 
the likehoold that a given student will pass, quantifying whether an intervention is necessary.
"""

# Import libraries
from __future__ import division
import pandas as pd
from time import time
from sklearn.metrics import f1_score
from sklearn import cross_validation

# Import the three supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import ensemble
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"
 
# Calculate number of students
n_students = len(student_data)
 
# Calculate number of features
n_features = len(student_data.columns) - 1 # -1 for target column
 
# Calculate passing students
n_passed = len(student_data[student_data.passed == "yes"])
 
# Calaculte failing students
n_failed = len(student_data[student_data.passed ==  "no"])
 
# Calculate graduation ratio
grad_rate = n_passed/n_students * 100
 
# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1]

# Show the lise of columns
print "Features columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Seperate the data into feature data and target data (X_all and Y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeautre values:"
print X_all.head()

def preprocess_features(X):
    '''Preprocesses the student data and converts non-numeric binary variable into
    binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)
    
    # Investigate each feature column for the data
    for col,col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/np values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes','no'],[1,0])
        
        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)
        
        # Collect the revised columns
        output = output.join(col_data)
        
    return output

def train_classifier(clf,X_train,y_train):
    ''' Fits a classifier to the training data. '''
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train,y_train)
    end = time()
    print "Trained model in {:.4f} seconds".format(end - start)

def predict_labels(clf,features,target):
    ''' Make predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end-start)
    return f1_score(target.values,y_pred, pos_label ='yes')
    
def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifier based on F1 score. '''
    
    # Indicated the classifier and the training set size
    print  "Training a {} using a training set size of {} ...".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))

# Test preprocess_features(X)
X_all = preprocess_features(X_all)
print "Processed feature column ({} total features):\n{}".format(len(X_all.columns),list(X_all.columns))

# Set the number of training points
num_train = 300

# Set the number of testing points
num_test = n_students - num_train

# Shuffle ans split the dataset into the number of training and testing points above
ss = cross_validation.ShuffleSplit(n_students,n_iter=1, test_size = num_test)
for train_index, test_index in ss:
    X_train = X_all.iloc[train_index]
    X_test = X_all.iloc[test_index]
    y_train = y_all.iloc[train_index]
    y_test = y_all.iloc[test_index]

# Show the results of the split
print "Training set has {} samples.".format(num_train)
print "Testing set has {} samples.".format(num_test)

# Initialize the three models
clf_A = ensemble.AdaBoostClassifier()
clf_B = GaussianNB()
clf_C = svm.SVC()

# Set up the training set sizes
X_train_100 = X_train[0:100]
y_train_100 = y_train[0:100]
X_train_200 = X_train[0:200]
y_train_200 = y_train[0:200]
X_train_300 = X_train[0:300]
y_train_300 = y_train[0:300]
print
# Execute the 'train_predict' function for each classifier and each training set size
# train_predict(clf, X_train, y_train, X_test, y_test)
train_predict(clf_A,X_train_100,y_train_100, X_test,y_test)
train_predict(clf_A,X_train_200,y_train_200, X_test,y_test)
train_predict(clf_A,X_train_300,y_train_300, X_test,y_test)
print 
train_predict(clf_B,X_train_100,y_train_100, X_test,y_test)
train_predict(clf_B,X_train_200,y_train_200, X_test,y_test)
train_predict(clf_B,X_train_300,y_train_300, X_test,y_test)
print
train_predict(clf_C,X_train_100,y_train_100, X_test,y_test)
train_predict(clf_C,X_train_200,y_train_200, X_test,y_test)
train_predict(clf_C,X_train_300,y_train_300, X_test,y_test)

# AdaBoost Model tuning 
# Create the parameters list you wish to tune
parameters = {'n_estimators':[20,30,40,50,60,70]}

# initialize the classifier
clf = clf_A

# Make an f1 scoriing function using 'make_scorer'
f1_scorer = make_scorer(f1_score, pos_label = 'yes')

# Perform grid search on the classifier using the f1_scorer as the  scoring method
grid_obj = GridSearchCV(clf, param_grid = parameters, scoring = f1_scorer)

# Fit the grid search object to the training data and find the optimal parameters
grid_obj.fit(X_train,y_train)

# Get the best tuned estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf,X_train,y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf,X_test,y_test))
print "Tuned model has an optimal parameter: ", grid_obj.best_params_
print "Features importances array is :", clf.feature_importances_
print "Key Features for identifying 'Pass/Fail' are:", X_all.columns[clf.feature_importances_>0.1]
