import pandas as pd
import numpy as np
import csv
import os
import pdb

train_address = 'C:\Users\\anike\Documents\Code\Titanic\Titanic Data\\train.csv'
train = pd.read_csv(train_address)
test_address = 'C:\Users\\anike\Documents\Code\Titanic\Titanic Data\\test.csv'
test = pd.read_csv(test_address)
result_address = 'C:\Users\\anike\Documents\Code\Titanic\Titanic Data\Results\\results.csv'

# Percentage survivals:
print 'Male: \n', train['Survived'][train['Sex'] == 'male'].value_counts(normalize = True)
print 'Female: \n', train['Survived'][train['Sex'] == 'female'].value_counts(normalize = True)

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
    
