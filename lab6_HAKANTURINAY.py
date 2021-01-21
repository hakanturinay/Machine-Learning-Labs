import csv
import numpy as np
import matplotlib.pyplot as plt

def simlin_r2(y, y_pred):

    rss, tss = 0, 0

    for i in range(len(y)):
        rss += (y[i]-y_pred[i])**2
        tss += (y[i]-np.mean(y))**2

    return 1 - rss/tss

def adjusted_r_squared(X,y):

    n = len(y)
    d = X.shape[1]
    adj_r = 1-(1-simlin_r2(y,y_hat))*((n-1)/(n-d-1))

    print("Adjusted R^2 score: " + str(adj_r))

with open('team_big.csv', encoding="utf8", errors='ignore') as f:
    csv_list = list(csv.reader(f))

age_list = np.array([])
exp_list = np.array([])
pow_list = np.array([])
salary_list = np.array([])

for row in csv_list:
    if row != csv_list[0]:
        age_list = np.append(age_list, int(row[4]))
        exp_list = np.append(exp_list, int(row[6]))
        pow_list = np.append(pow_list, float(row[7]))
        salary_list = np.append(salary_list, int(row[8]))

ones = np.ones((len(age_list)))
X = np.column_stack((ones, age_list, exp_list, pow_list))
y = salary_list
coefficients = np.dot(X.T, X)
coefficients = np.linalg.inv(coefficients)
coefficients = np.dot(coefficients, X.T)
coefficients = np.dot(coefficients, y)
y_hat = np.dot(X,coefficients)

x0 = X[:,0] #Ones
x1 = X[:,1] #Age
x2 = X[:,2] #Experience
x3 = X[:,3] #Power
rand_col = np.random.randint(-1000,1000, len(age_list))
x4 = np.column_stack((X, rand_col))

m4 = adjusted_r_squared(x4,y)
m3 = adjusted_r_squared(X[:,[0,1,2,3]],y)
m2 = adjusted_r_squared(X[:,[1,2]],y)
m1 = adjusted_r_squared(X[:,[1]],y)
m0 = adjusted_r_squared(X[:,[0]],y)

