import pandas as pd
import numpy as np

filename1 = 'prices_round_1_day_-2.csv'
filename2 = 'prices_round_1_day_-1.csv'
filename3 = 'prices_round_1_day_0.csv'

## Preprocessing

file1 = pd.read_csv(filename1, sep=';')  ## The used separator is ;
file1 = file1[file1['product'] == 'STARFRUIT']  ## Only use amethyst data
file1 = file1.filter(['mid_price', 'timestamp'])  ## Only use the midprice and the timestamp column

## Same for the other files
file2 = pd.read_csv(filename2, sep=';')
file2 = file2[file2['product'] == 'STARFRUIT']
file2 = file2.filter(['mid_price', 'timestamp'])

file3 = pd.read_csv(filename3, sep=';')
file3 = file3[file3['product'] == 'STARFRUIT']
file3 = file3.filter(['mid_price', 'timestamp'])

file2['timestamp'] = file2['timestamp'].apply(lambda x: x + 1000000)
file3['timestamp'] = file3['timestamp'].apply(lambda x: x + 2000000)

data = pd.concat([file1, file2, file3])  ## Three days merged into 1 frame

###Linear regression setup
n = 15  ## This is the amount of explainatory variables
k = 1  ## This is how much in the future we want to look ## not yet implemented

# design matrix, this can probably be done nicer
vals = data['mid_price'].to_numpy()
Y = vals[n:]
X = np.ones((30000 - n, n + 1))
for i in range(len(Y)):
    for j in range(n):
        X[i][j + 1] = vals[i + j]

beta = np.linalg.lstsq(X, Y)
print(beta[0])  ## Those are the regression parameters, The first is the constant term ('intercept' in the code).