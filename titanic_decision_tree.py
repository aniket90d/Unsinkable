import pandas as pd
import numpy as np
import csv
import os
import pdb
from sklearn import tree

pd.options.mode.chained_assignment = None  

# Functions:
def get_true_positives(prediction, truth):
    len_TP = 0
    for i, x in enumerate(prediction):
        if prediction[i] == 1 and truth[i] == 1:
            len_TP += 1
    return len_TP

def get_true_negatives(prediction, truth):
    len_TN = 0
    for i, x in enumerate(prediction):
        if prediction[i] == 0 and truth[i] == 0:
            len_TN += 1
    return len_TN

#===========================================================================================================

# Load test and train data
train_address = 'C:\Users\\anike\Documents\Code\Titanic\Titanic Data\\train.csv'
train = pd.read_csv(train_address)
test_address = 'C:\Users\\anike\Documents\Code\Titanic\Titanic Data\\test.csv'
test = pd.read_csv(test_address)
result_address = 'C:\Users\\anike\Documents\Code\Titanic\Titanic Data\Results\\results.csv'

# Clean up train
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Sex'][train['Sex'] == 'male'] = 0
train['Sex'][train['Sex'] == 'female'] = 1
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].value_counts().index.tolist()[0])
train['Embarked'][train['Embarked'] == 'S'] = 0
train['Embarked'][train['Embarked'] == 'C'] = 1
train['Embarked'][train['Embarked'] == 'Q'] = 2

# Clean up test
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test['Sex'][test['Sex'] == 'male'] = 0
test['Sex'][test['Sex'] == 'female'] = 1
test['Embarked'] = test['Embarked'].fillna(test['Embarked'].value_counts().index.tolist()[0])
test['Embarked'][test['Embarked'] == 'S'] = 0
test['Embarked'][test['Embarked'] == 'C'] = 1
test['Embarked'][test['Embarked'] == 'Q'] = 2

# Features to include
train['family_size'] = train['SibSp'] + train['Parch'] + 1
test['family_size'] = test['SibSp'] + test['Parch'] + 1
train['is_child'] = 0
train['is_child'][train['Age'] < 18] = 1
test['is_child'] = 0
test['is_child'][test['Age'] < 18] = 1
include_features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked', 'family_size', 'is_child']

# Set decision tree parameters:
set_default = False
if set_default:
    max_depth = None
    min_samples_split = 2
else:
    max_depth = 10
    min_samples_split = 50

# Cross-validation
cv_len = 2 * len(train['Survived']) / 3
target_train = train['Survived'][:cv_len]
target_cv = train['Survived'][cv_len:]
target_cv = target_cv.tolist() # converted so that indexing begins from 0
features_train = train[include_features][:cv_len]
features_cv = train[include_features][cv_len:]

tree_cv = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split)
tree_cv = tree_cv.fit(features_train, target_train)
predicted_train = tree_cv.predict(features_train)
predicted_cv = tree_cv.predict(features_cv)

train_TP = get_true_positives(predicted_train, target_train)
train_TN = get_true_negatives(predicted_train, target_train)
cv_TP = get_true_positives(predicted_cv, target_cv)
cv_TN = get_true_negatives(predicted_cv, target_cv)

train_acc = (train_TP + train_TN) * 100.0 / cv_len
cv_acc = (cv_TP + cv_TN) * 100.0 / (len(train['Survived']) - cv_len)

print('Training Accuracy: {:{width}.{prec}f}%'.format(train_acc, width = 5, prec = 2))
print('Test Accuracy (CV): {:{width}.{prec}f}%'.format(cv_acc, width = 5, prec = 2))
                 
# Create target and feature arrays:
target = train['Survived'].values
train_features = train[include_features].values
test_features = test[include_features].values

# Fit decision tree
my_tree = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split)
my_tree = my_tree.fit(train_features, target)

# Predict using decision tree:
test_dt = test
test_dt['Survived'] = my_tree.predict(test_features)

# Save results file
result = np.transpose([np.array(test_dt['PassengerId']), np.array(test_dt['Survived'])])
# print 'Result: \n', result

if (os.path.exists(result_address)):
    with open(result_address, 'ab') as f:
        f.truncate()
        
    f = open(result_address, 'wb')
    header = ['PassengerId', ',', 'Survived', ',', '\n']
    [f.write(header[i]) for i in range(len(header))]
    f.close()
    
    with open(result_address, 'ab') as f:
        writer = csv.writer(f)
        [writer.writerow(result[i, :]) for i in range(len(result))]
else:
    f = open(result_address, 'wb')
    header = ['PassengerId', ',', 'Survived', ',', '\n']
    [f.write(header[i]) for i in range(len(header))]
    f.close()
    with open(result_address, 'ab') as f:
        writer = csv.writer(f)
        [writer.writerow(result[i, :]) for i in range(len(result))]
    
