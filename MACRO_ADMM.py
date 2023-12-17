import csv
import numpy as np
from scipy.optimize import minimize

# Flag_for_first_run
#flag_first = True
flag_first = False

### Input Dataset:
# resolution_accuracy_pedestrians_nearby, by default
a_11 = 1.0056546333464234
b_11 = 0.3427494323910407
c_11 = 298.12356779186206
# resolution_accuracy_pedestrians_faraway
a_12 = 0.9909859254850617
b_12 = 4.256426115928811
c_12 = 205.4534825948463
# frame_rate_accuracy_low_cars, by default
a_21 = 0.9909265007048699
b_21 = 0.3294484223183229
c_21 = 3.5893107024055504
# frame_rate_accuracy_fast_cars
a_22 = 1.0480839254828827
b_22 = 1.042089038891055
c_22 = 9.80570183959145

# Platform Number
N = 4
# Worker Number
M = 10

# Parameter Setting:
Fn_array = [i for i in range(N)]

re_array = [j for j in range(M)]
resolution_set = [240, 360, 480, 540, 640, 720, 800, 960, 1080]

a_re_array = np.zeros((N, M))
b_re_array = np.zeros((N, M))
c_re_array = np.zeros((N, M))
a_fr_array = np.zeros((N, M))
b_fr_array = np.zeros((N, M))
c_fr_array = np.zeros((N, M))

G_array = [i for i in range(N)]

b_array = np.zeros((N, M))
B_array = [i for i in range(N)]

gamma_d_array = [i for i in range(M)]
gamma_c_array = np.zeros((N,M))
omega_array = [i for i in range(M)]

c_array = np.zeros((N, M))
C_array = [i for i in range(M)]

# Fn:
data = []
with open('./dataset/Fn_parameter.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append([float(value) for value in row])
for i in range(N):
    Fn_array[i] = data[i][0]
# re:
data = []
with open('./dataset/re_parameter.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append([float(value) for value in row])
for j in range(M):
    re_array[j] = data[j][0]
# video analytics:
data = []
with open('./dataset/accuracy_parameter.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append([float(value) for value in row])
for i in range(N):
    for j in range(M):
        a_re_array[i, j] = data[i * M + j][0]
        b_re_array[i, j] = data[i * M + j][1]
        c_re_array[i, j] = data[i * M + j][2]
        a_fr_array[i, j] = data[i * M + j][3]
        b_fr_array[i, j] = data[i * M + j][4]
        c_fr_array[i, j] = data[i * M + j][5]
# Profit Gn:
data = []
with open('./dataset/G_parameter.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append([float(value) for value in row])
for i in range(N):
    G_array[i] = data[i][0]
# b_n,m:
data = []
with open('./dataset/b_parameter.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append([float(value) for value in row])
for i in range(N):
    for j in range(M):
        b_array[i, j] = data[i * M + j][0]
# B:
data = []
with open('./dataset/BB_parameter.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append([float(value) for value in row])
for i in range(N):
    B_array[i] = data[i][0]
# gamma_d:
data = []
with open('./dataset/gamma_d_parameter.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append([float(value) for value in row])
for j in range(M):
    gamma_d_array[j] = data[j][0]
# gamma_c
data = []
with open('./dataset/gamma_c_parameter.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append([float(value) for value in row])
for i in range(N):
    for j in range(M):
        gamma_c_array[i, j] = data[i * M + j][0]
# omega:
data = []
with open('./dataset/omega_parameter.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append([float(value) for value in row])
for j in range(M):
    omega_array[j] = data[j][0]
# c_n,m:
data = []
with open('./dataset/c_parameter.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append([float(value) for value in row])
for i in range(N):
    for j in range(M):
        c_array[i, j] = data[i * M + j][0]
# C:
data = []
with open('./dataset/CC_parameter.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append([float(value) for value in row])
for j in range(M):
    C_array[j] = data[j][0]

# Define a optimization problem for each platform:
# Utility:
def accuracy_func(x, a, b, c):
    return a - b * np.exp(-1 * x / c)


result_res = accuracy_func(480, a_11, b_11, c_11)
result_fr = accuracy_func(30, a_21, b_21, c_21)
print(result_res * result_fr)

def Energy(x, n, m):
    return omega_array[m] * (gamma_d_array[m] * re_array[m] * re_array[m]  + gamma_c_array[n][m]) * x


def objective_function(x, theta):
    X_matrix = x.reshape(N, M)
    result = 0
    for i in range(N):
        for j in range(M):
            result = result + G_array[i] * \
                     accuracy_func(X_matrix[i][j], a_fr_array[i, j], b_fr_array[i, j], c_fr_array[i, j]) * \
                     accuracy_func(re_array[j], a_re_array[i, j], b_re_array[i, j], c_re_array[i, j]) + \
                     Energy(X_matrix[i][j], i, j) - theta[i,j] * X_matrix[i][j]
            #result = result + G_array[i] * X_matrix[i][j]
    return result * (-1)


def platforms_generate_constraint(index):
    def constraint(x):
        X_matrix = x.reshape(N, M)
        result = 0
        for j in range(M):
            result = result + X_matrix[index][j] * b_array[index, j]
        return B_array[index] - result
    return constraint


def workers_generate_constraint(index):
    def constraint(x):
        X_matrix = x.reshape(N, M)
        result = 0
        for i in range(N):
            result = result + X_matrix[i][index] * c_array[i, index]
        return C_array[index] - result
    return constraint


def Fn_generate_inequality_constraint(index):
    def constraint(x):
        X_matrix = x.reshape(N, M)
        result = 0
        for i in range(N):
            for j in range(M):
                if i * M + j == index:
                    return Fn_array[i] - X_matrix[i][j]
    return constraint


def zero_generate_inequality_constraint(index):
    def constraint(x):
        X_matrix = x.reshape(N, M)
        result = 0
        for i in range(N):
            for j in range(M):
                if i * M + j == index:
                    return X_matrix[i][j] - 0
    return constraint


def constraint_0(x):
    X_matrix = x.reshape(N, M)
    return 5 - X_matrix[0][0]


def callback_function(theta):
    def callback(xk):
        print("Current optimization vector:", xk)
        print("Current objective value:", objective_function(xk, theta))
        print()
        # strategy
        row_strategy = []
        rows_strategy = []
        for i in range(N):
            for j in range(M):
                row_strategy.append(xk[i * M + j])
        rows_strategy.append(row_strategy)
        with open('./result/strategy_ADMM.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows_strategy)
        # accuracy
        row_accuracy = []
        rows_accuracy = []
        for i in range(N):
            for j in range(M):
                row_accuracy.append(accuracy_func(xk[i * M + j], a_fr_array[i, j], b_fr_array[i, j], c_fr_array[i, j]) * \
                accuracy_func(re_array[j], a_re_array[i, j], b_re_array[i, j], c_re_array[i, j]))
        rows_accuracy.append(row_accuracy)
        with open('./result/accuracy_ADMM.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows_accuracy)
        # utility
        row_utility = []
        rows_utility = []
        for i in range(N):
            for j in range(M):
                row_utility.append(G_array[i] * accuracy_func(xk[i * M + j], a_fr_array[i, j], b_fr_array[i, j], c_fr_array[i, j]) * \
                                    accuracy_func(re_array[j], a_re_array[i, j], b_re_array[i, j], c_re_array[i, j]) + \
                                    Energy(xk[i * M + j], i, j) - theta[i,j] * xk[i * M + j])
        rows_utility.append(row_utility)
        with open('./result/utility_ADMM.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows_utility)
    return callback

def write_random_initial(xk, theta):
    # strategy
    row_strategy = []
    rows_strategy = []
    for i in range(N):
        for j in range(M):
            row_strategy.append(xk[i * M + j])
    rows_strategy.append(row_strategy)
    with open('./result/strategy_ADMM.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows_strategy)
    # accuracy
    row_accuracy = []
    rows_accuracy = []
    for i in range(N):
        for j in range(M):
            row_accuracy.append(accuracy_func(xk[i * M + j], a_fr_array[i, j], b_fr_array[i, j], c_fr_array[i, j]) * \
                                accuracy_func(re_array[j], a_re_array[i, j], b_re_array[i, j], c_re_array[i, j]))
    rows_accuracy.append(row_accuracy)
    with open('./result/accuracy_ADMM.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows_accuracy)
    # utility
    row_utility = []
    rows_utility = []
    for i in range(N):
        for j in range(M):
            row_utility.append(
                G_array[i] * accuracy_func(xk[i * M + j], a_fr_array[i, j], b_fr_array[i, j], c_fr_array[i, j]) * \
                accuracy_func(re_array[j], a_re_array[i, j], b_re_array[i, j], c_re_array[i, j]) + \
                Energy(xk[i * M + j], i, j) - theta[i, j] * xk[i * M + j])
    rows_utility.append(row_utility)
    with open('./result/utility_ADMM.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows_utility)

theta = np.zeros((N,M))
x = np.zeros((N, M))
for i in range(N):
    for j in range(M):
        x[i, j] = np.random.rand() * Fn_array[i]

print(x)

x0 = x.flatten()
write_random_initial(x0, theta)


my_constraints = []
num_constraints = N
my_constraints.extend([{'type': 'ineq', 'fun': platforms_generate_constraint(i)} for i in range(num_constraints)])
num_constraints = M
my_constraints.extend([{'type': 'ineq', 'fun': workers_generate_constraint(j)} for j in range(num_constraints)])
num_constraints = N * M
my_constraints.extend([{'type': 'ineq', 'fun': Fn_generate_inequality_constraint(k)} for k in range(num_constraints)])
my_constraints.extend([{'type': 'ineq', 'fun': zero_generate_inequality_constraint(k)} for k in range(num_constraints)])
#my_constraints.extend([{'type': 'ineq', 'fun': constraint_0}])

result = minimize(objective_function, x0, args=(theta), constraints=my_constraints, callback=callback_function(theta))

optimal_X = result.x.reshape(N, M)

print("Optimal X:")
print(optimal_X)
print("Optimal Objective:")
print(result.fun)