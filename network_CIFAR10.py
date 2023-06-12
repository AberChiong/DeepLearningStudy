'''

我将此作为一个阶段性练习
用于检测自己这个阶段对于卷积神经网络的掌握程度
本模型采用pytorch框架
CIFAR10数据集
将依次实现
于cpu上运算2个epoch并记录时间
于cuda上运算2个epoch并记录时间
分别对两个模型时间比较和准确率的比较

'''


# 0 模块的引入
import sys
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from functools import wraps

# 0.1 程序初始化设置 允许反复调取动态库
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 0.2 简单函数
# 0.2.1 展示图片的函数
def imshow(img):
    img = img / 2 + 0.5     # 非归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 0.2.2 训练结果展示
def rsshow(net):
    # 随机获取测试集的一个 batch
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    # 图片和分类展示
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # 网络输出
    device = str(next(net.parameters()).device)
    if device == "cuda:0":
        images, labels = images.cuda(), labels.cuda()
    
    outputs = net(images)
    # 预测结果
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# 0.2.3 模型准确度测试
def mdacc(net): # 总准确度
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            device = str(next(net.parameters()).device)
            if device == "cuda:0":
                images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = acc = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % acc)
    return acc

def mdacc_sin(net): # 分类准确度
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            device = str(next(net.parameters()).device)
            if device == "cuda:0":
                images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))



# 0.3 装饰器
# 0.3.1 记录方法执行耗时
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = int(time.time() - start) # 统计耗时
        print('method: %s, time: %s' % (func.__name__, duration))
        return result
    return wrapper


# 1 数据集的获取 & 数据归一化
def get_data_loaders(batch_size=4, num_workers=0):
    # 定义数据变换
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # 获取训练集和测试集
    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
    trainloader = data.DataLoader(trainset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers)

    testset = datasets.CIFAR10(root='./data', train=False,
                            download=True, transform=transform)
    testloader = data.DataLoader(testset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)

    # 定义类别标签
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


# 2 构建网络
# 2.1 网络模型类和正向传播

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(64, 10)
 
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

# 2.2 设置交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 2.3 设置启动带有动量的SDG优化器
class Opt():
    def __init__(self, net):
        self.net = net

    def optimizer(self, ln, av):
        optimizer = optim.SGD(self.net.parameters(),
                              lr = ln, momentum = av)
        return optimizer
    
# 3 反向传播 训练开始
# 3.1.1 反向传播

@timeit
def train(device, epoch, net, opt):
    for i in range(epoch):
        running_loss = 0.0
        for j, data in enumerate(trainloader, 0):
            # 数据的获取和输入
            inputs, labels = data
            # 区分运算位置
            if device == device_gpu:
                inputs, labels = inputs.to(device), labels.to(device)
            # 清空梯度
            opt.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

                # 打印统计信息
            running_loss += loss.item()
            if j % 2500 == 2499:
                # 每 2500 次迭代打印一次信息
                print('[%d, %5d] loss: %.3f' % (i + 1, j + 1, running_loss / 2500))
                running_loss = 0.0
    return net


if __name__ == '__main__':
    # 3.1.2 检测环境
    if torch.cuda.is_available() == True:
        print("cuda 正常，程序继续运行。")
        device_cpu = torch.device("cpu")
        device_gpu = torch.device("cuda:0")
    else:
        print("cuda 异常，请检测显卡运行状况，程序已经退出。")
        sys.exit(0)

    # 3.1.3 获取数据
    trainloader, testloader, classes=get_data_loaders()
    
    # 3.2 CPU 训练
    print('#'* 50)
    print("CPU 训练开始！")

    net_cpu = Net()
    opt_cpu = Opt(net_cpu).optimizer(0.001, 0.9)

    net_cpu_tr = train(device_cpu, epoch=1, net=net_cpu, opt=opt_cpu)

    print("CPU 训练结果展示：")
    rsshow(net_cpu_tr)
    cpu_acc = mdacc(net_cpu_tr)
    print("分型预测准确度如下：")
    mdacc_sin(net_cpu_tr)

    print("恭喜你，CPU模型训练完全结束！模型的准确度为：%s" % cpu_acc)
    # 保存模型
    torch.save(net_cpu_tr, 'model_cpu.pth')

    # 3.3 GPU 训练
    print('#'* 50)
    print("GPU 训练开始！")

    net_gpu = Net().to(device_gpu)
    opt_gpu = Opt(net_gpu).optimizer(0.001, 0.9)

    # 训练模型
    net_gpu_tr = train(device_gpu, epoch=1, net=net_gpu, opt=opt_gpu)

    print("GPU 训练结果展示：")
    rsshow(net_gpu_tr)
    
    gpu_acc = mdacc(net_gpu_tr)
    print("分型预测准确度如下：")
    mdacc_sin(net_gpu_tr)
    
    print("恭喜你，GPU模型训练完全结束！模型的准确度为：%s" % gpu_acc)
    # 保存模型
    torch.save(net_gpu_tr, 'model_gpu.pth')
    
