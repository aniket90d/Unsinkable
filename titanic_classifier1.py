import pandas as pd
import numpy as np
import csv
import os
import pdb

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

train_address = 'C:\Users\\anike\Documents\Code\Titanic\Titanic Data\\train.csv'
train = pd.read_csv(train_address)
test_address = 'C:\Users\\anike\Documents\Code\Titanic\Titanic Data\\test.csv'
test = pd.read_csv(test_address)
result_address = 'C:\Users\\anike\Documents\Code\Titanic\Titanic Data\Results\\results.csv'

# Percentage survivals:
print 'Male: \n', train['Survived'][train['Sex'] == 'male'].value_counts(normalize = True)
print 'Female: \n', train['Survived'][train['Sex'] == 'female'].value_counts(normalize = True)

# Cross-validation
cv_len = 2 * len(train['Survived']) / 3
target_train = train['Survived'][:cv_len]
target_cv = train['Survived'][cv_len:]
target_cv = target_cv.tolist() # converted so that indexing begins from 0
train['Predicted'] = 0
train['Predicted'][train['Sex'] == 'female'] = 1
predicted_train = train['Predicted'][:cv_len]
predicted_cv = train['Predicted'][cv_len:]
predicted_cv = predicted_cv.tolist() # converted so that indexing begins from 0

train_TP = get_true_positives(predicted_train, target_train)
train_TN = get_true_negatives(predicted_train, target_train)
cv_TP = get_true_positives(predicted_cv, target_cv)
cv_TN = get_true_negatives(predicted_cv, target_cv)

train_acc = (train_TP + train_TN) * 100.0 / cv_len
cv_acc = (cv_TP + cv_TN) * 100.0 / (len(train['Survived']) - cv_len)

print('Training Accuracy: {:{width}.{prec}f}%'.format(train_acc, width = 5, prec = 2))
print('Test Accuracy (CV): {:{width}.{prec}f}%'.format(cv_acc, width = 5, prec = 2))

# Predict based on gender:
test_gender = test
test_gender['Survived'] = 0
test_gender['Survived'][test_gender['Sex'] == 'female'] = 1

# Save results file
result = np.transpose([np.array(test_gender['PassengerId']), np.array(test_gender['Survived'])])
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
    
