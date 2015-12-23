__author__ = 'Horace'

import numpy as np, pandas as pd
from sklearn import linear_model
train_data = pd.read_csv("train_data.csv")

# data visualization
import matplotlib.pyplot as plt

for i in range(2,5):
    plt.subplot(2,2,i-1)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.scatter(train_data[[i]], train_data.consumption_amount, color="black")
    plt.xlabel(train_data.columns[i])
    plt.ylabel(train_data.columns[5])

plt.show()

# split the data into input and target variables
train_input = np.array(train_data[[i for i in range(2,5)]])
train_target = np.array(train_data[[5]])

model = linear_model.LinearRegression()  # Default: fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
model.fit(train_input, train_target)

coef_of_det = model.score(train_input, train_target)  # score is the Coefficient of Determination
sse_train = sum((model.predict(train_input) - train_target)**2)
print "Intercept: ", model.intercept_
print "Coefficient: ", model.coef_
print "Sum of Square Error of Training Data: ", sse_train
print "R square: ", coef_of_det


# predict the testing data
test_data = pd.read_csv("test_data.csv")
test_input = np.array(test_data[[i for i in range(2,5)]])
test_target = model.predict(test_input)

for i in range(5):
    print test_data.name[i], ": ", test_target[i][0]
