import numpy as np

# 创建4维张量
tensor4d = np.zeros((3, 4, 2, 5))
print(tensor4d)

# 在索引(1, 2, 1, 3)处设置值为1
tensor4d[1, 2, 1, 3] = 1

# 访问索引(1, 2, 1, 3)处的值
print(tensor4d)

import numpy as np

# 创建一个大小为32x32x32的随机张量数据
tensor_data = np.random.rand(32, 32, 32)

# 输出张量的形状和数据类型
print('Tensor shape:', tensor_data.shape)
print('Tensor data type:', tensor_data.dtype)

tensor_51 = np.random.rand(5, 1)
tensor_15 = np.random.rand(1, 5)

print(tensor_51)
print(tensor_15)