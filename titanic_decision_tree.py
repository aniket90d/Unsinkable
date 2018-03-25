import pandas as pd
import numpy as np
import csv
import os
import pdb
from sklearn import tree

pd.options.mode.chained_assignment = None  

# Load test and train data
train_address = 'C:\Users\\anike\Documents\Code\Titanic\Titanic Data\\train.csv'
train = pd.read_csv(train_address)
test_address = 'C:\Users\\anike\Documents\Code\Titanic\Titanic Data\\test.csv'
test = pd.read_csv(test_address)
result_address = 'C:\Users\\anike\Documents\Code\Titanic\Titanic Data\Results\\results.csv'

# Clean up train
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].value_counts().index.tolist()[0])
train['Sex'][train['Sex'] == 'male'] = 0
train['Sex'][train['Sex'] == 'female'] = 1
train['Embarked'][train['Embarked'] == 'S'] = 0
train['Embarked'][train['Embarked'] == 'C'] = 1
train['Embarked'][train['Embarked'] == 'Q'] = 2

# Clean up test
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test['Sex'][test['Sex'] == 'male'] = 0
test['Sex'][test['Sex'] == 'female'] = 1

# Cross-validation
cv_len = 2 * len(train['Survived']) / 3
target_train = train['Survived'][:cv_len]
target_cv = train['Survived'][cv_len:]
features_train = train[['Pclass', 'Sex', 'Age', 'Fare']][:cv_len]
features_cv = train[['Pclass', 'Sex', 'Age', 'Fare']][cv_len:]

tree_cv = tree.DecisionTreeClassifier()
tree_cv = tree_cv.fit(features_train, target_train)
predicted_train = tree_cv.predict(features_train)
predicted_cv = tree_cv.predict(features_cv)

train_acc = 100 - np.sum(np.absolute(predicted_train - target_train)) * 100.0 / cv_len
cv_acc = 100 - np.sum(np.absolute(predicted_cv - target_cv)) * 100.0 / (len(train['Survived']) - cv_len)

print('Training Accuracy: {:{width}.{prec}f}%'.format(train_acc, width = 5, prec = 2))
print('Test Accuracy (CV): {:{width}.{prec}f}%'.format(cv_acc, width = 5, prec = 2))
                 
# Create target and feature arrays:
target = train['Survived'].values
train_features = train[['Pclass', 'Sex', 'Age', 'Fare']].values
test_features = test[['Pclass', 'Sex', 'Age', 'Fare']].values

# Fit decision tree
my_tree = tree.DecisionTreeClassifier()
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
    
