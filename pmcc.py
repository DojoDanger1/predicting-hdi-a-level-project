import csv
import numpy as np
import seaborn

# load the training data
with open('data/training_data.csv', 'r') as file:
    reader = csv.DictReader(file)
    trainingData = []
    for record in reader:
        trainingData.append(record)

# iterate over every pair of factors
factors = list(trainingData[0].keys())[3:]
pmcc_grid = []
for factor1 in factors:
    pmcc_row = []
    for factor2 in factors:
        x_values = [record[factor1] for record in trainingData]
        y_values = [record[factor2] for record in trainingData]
        print(f'calculating pmcc of {factor1} and {factor2}...') # log message
        # remove pairs with empty values, iterating over the lists in reverse as to not disturb the index numbers
        for index in range(len(x_values)-1, -1, -1):
            if x_values[index] == '' or y_values[index] == '':
                x_values.pop(index)
                y_values.pop(index)
        # convert to float
        x_values = [float(x) for x in x_values]
        y_values = [float(y) for y in y_values]
        # calculate pmcc and add it to grid
        pmcc = sum([(x_values[index]-np.mean(x_values))*(y_values[index]-np.mean(y_values)) for index in range(len(x_values))])/np.sqrt((sum([(x_values[index]-np.mean(x_values))**2 for index in range(len(x_values))]))*(sum([(y_values[index]-np.mean(y_values))**2 for index in range(len(x_values))])))
        pmcc_row.append(pmcc)
    pmcc_grid.append(pmcc_row)

# generate the image
heatmap = seaborn.heatmap(pmcc_grid, xticklabels=factors, yticklabels=factors, vmin=-1, vmax=1, center=0, cmap='Spectral')
heatmap.get_figure().savefig('data/output.png', bbox_inches='tight', dpi=600)