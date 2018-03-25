import pandas as pd
import numpy as np
import csv
import os
import pdb

pd.options.mode.chained_assignment = None

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
train['Predicted'] = 0
train['Predicted'][train['Sex'] == 'female'] = 1
predicted_train = train['Predicted'][:cv_len]
predicted_cv = train['Predicted'][cv_len:]

train_acc = 100 - np.sum(np.absolute(predicted_train - target_train)) * 100.0 / cv_len
cv_acc = 100 - np.sum(np.absolute(predicted_cv - target_cv)) * 100.0 / (len(train['Survived']) - cv_len)

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
    
