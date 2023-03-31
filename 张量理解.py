import numpy as np

# 创建4维张量
tensor4d = np.zeros((3, 4, 2, 5))
print(tensor4d)

# 在索引(1, 2, 1, 3)处设置值为1
tensor4d[1, 2, 1, 3] = 1

# 访问索引(1, 2, 1, 3)处的值
print(tensor4d)
