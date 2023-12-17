import numpy as np
from scipy.optimize import minimize
import csv

N = 4
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

def accuracy_func(x, a, b, c):
    return a - b * np.exp(-1 * x / c)



# 定义目标函数和约束函数
def objective_function(x, theta, m, n, theta_new):
    X_matrix = x.reshape(N, M)
    result = 0
    for i in range(N):
        for j in range(M):
            result = result + G_array[i] * \
                    accuracy_func(X_matrix[i][j], a_fr_array[i, j], b_fr_array[i, j], c_fr_array[i, j]) * \
                     accuracy_func(re_array[j], a_re_array[i, j], b_re_array[i, j], c_re_array[i, j])
            #result = result + G_array[i] * X_matrix[i][j]
    result = result + (theta[m][n]- theta_new) * X_matrix[m][n]
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


def constraint_function_fix(x, fixed_value, n, m):
    X_matrix = x.reshape(N, M)
    return X_matrix[n][m] - fixed_value

platform_worker = [0,1,2,3]
step_size = 0.1

# calculate the optimal theta
theta = np.zeros((N,M))
for i in range(N):
    for j in range(M):
        theta[i][j] = (gamma_d_array[j] * re_array[j] * re_array[j] + gamma_c_array[i][j]) * omega_array[j]

candidate_theta = [[],[],[],[]]
for i in range(4):
    for j in range(20):
        candidate_theta[i].append((j - 10) * step_size + theta[i][platform_worker[i]])

rows_profit = []

for platform in range(N):
    # 进行优化
    row_profit = []
    for fixed_value in candidate_theta[platform]:
        # 初始猜测值
        x = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                x[i, j] = np.random.rand() * Fn_array[i]
        x0 = x.flatten()

        my_constraints = []
        num_constraints = N
        my_constraints.extend(
            [{'type': 'ineq', 'fun': platforms_generate_constraint(i)} for i in range(num_constraints)])
        num_constraints = M
        my_constraints.extend([{'type': 'ineq', 'fun': workers_generate_constraint(j)} for j in range(num_constraints)])
        num_constraints = N * M
        my_constraints.extend(
            [{'type': 'ineq', 'fun': Fn_generate_inequality_constraint(k)} for k in range(num_constraints)])
        my_constraints.extend(
            [{'type': 'ineq', 'fun': zero_generate_inequality_constraint(k)} for k in range(num_constraints)])
        #my_constraints.extend([{'type': 'eq', 'fun': lambda x: constraint_function_fix(x, fixed_value, n=platform, m=platform_worker[platform])}])

        # 使用 minimize 函数进行优化
        result = minimize(objective_function, x0, args=(theta, platform, platform_worker[platform], fixed_value),
                          constraints=my_constraints)

        # 获取优化结果
        optimized_variables = result.x
        print(f"Optimized variables with fixed value {fixed_value}:\n", optimized_variables.reshape(N, M))
        print(f"Sum of optimized variables with fixed value {fixed_value}: {result.fun}")

        r = 0
        for i in range(N):
            for j in range(M):
                r = r + G_array[i] * \
                         accuracy_func((result.x)[i*M+j], a_fr_array[i, j], b_fr_array[i, j], c_fr_array[i, j]) * \
                         accuracy_func(re_array[j], a_re_array[i, j], b_re_array[i, j], c_re_array[i, j])

        row_profit.append(-1 * r)
    rows_profit.append(row_profit)

with open('./result/MSE_platform.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(rows_profit)