import math
import random

# 定义函数
def f(x):
    return x ** 4 + 2 * x ** 3 - 3 * x ** 2 - 2 * x

# 定义模拟退火算法
def simulated_annealing(x0, T0, alpha, stopping_T, stopping_iter):
    x_best = x0
    f_best = f(x_best)
    x_curr = x_best
    f_curr = f_best
    T = T0
    i = 0
    while T > stopping_T and i < stopping_iter:
        x_new = x_curr + random.uniform(-1, 1)
        f_new = f(x_new)
        delta = f_new - f_curr
        if delta < 0 or math.exp(-delta / T) > random.uniform(0, 1):
            x_curr = x_new
            f_curr = f_new
        if f_curr < f_best:
            x_best = x_curr
            f_best = f_curr
        T *= alpha
        i += 1
    return x_best

# 设置初始值、初始温度、降温速率和停止条件
x0 = 1
T0 = 100
alpha = 0.95
stopping_T = 1e-8
stopping_iter = 1000

# 调用模拟退火算法求解最小值点
x_min = simulated_annealing(x0, T0, alpha, stopping_T, stopping_iter)

# 计算最小值
y_min = f(x_min)

print("xmin=", x_min)
print("ymin=", y_min)