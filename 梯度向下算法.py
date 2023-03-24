def gradient_descent(alpha, x, iterations):
    for i in range(iterations):
        x = x - alpha * (2*x - 2)
    return x

# 设置初始值、学习率和迭代次数
x0 = 5
alpha = 0.1
iterations = 100

# 调用梯度下降算法求解最小值点
x_min = gradient_descent(alpha, x0, iterations)

# 计算最小值
y_min = x_min ** 2 - 2 * x_min

# print("函数的最小值为：", y_min)
# print("最小值点为：", x_min)



import numpy as np
import matplotlib.pyplot as plt



def gradient_descent(alpha, x, iterations):
    for i in range(iterations):
        f = x ** 4 + 2 * x ** 3 - 3 * x ** 2 - 2 * x
        df = 4 * x ** 3 + 6 * x ** 2 - 6 * x - 2
        x -= alpha * df
    return x

# 设置初始值、学习率和迭代次数
x0 = -0.5
alpha = 0.01
iterations = 1000

# 调用梯度下降算法求解最小值点
x_min = gradient_descent(alpha, x0, iterations)

# 计算最小值
y_min = x_min ** 4 + 2 * x_min ** 3 - 3 * x_min ** 2 - 2 * x_min

print("函数的最小值为：", y_min)
print("最小值点为：", x_min)



# 定义函数
def f(x):
    return x ** 4 + 2 * x ** 3 - 3 * x ** 2 - 2 * x

# 绘制函数图像
x = np.linspace(-2.5, 1.5, 100)
y = f(x)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function Graph')
plt.show()

