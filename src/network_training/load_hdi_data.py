import csv
import numpy as np
import random
import pickle

# read csv file
with open('data/training_data.csv', 'r') as file:
    reader = csv.DictReader(file)
    hdiData = []
    for record in reader:
        # convert it to the right format
        hdiData.append({
            "input": np.matrix([[100 if factor == '' else float(factor)/10 if index <= 11 else float(factor)] for index, factor in enumerate(list(record.values())[4:])]),
            "output": np.matrix([[float(record['hdi'])]])
        })

# split the data set into training and testing
def test_train_split(data_set, proportion_train):
    random.shuffle(data_set)
    return {
        "training": data_set[:int((len(data_set)*proportion_train)//1)],
        "testing": data_set[int((len(data_set)*proportion_train)//1):]
    }

hdiData = test_train_split(hdiData, 0.9)
# save data to a file
with open(f'data/hdi.pkl', 'wb') as file:
    pickle.dump(hdiData, file)
