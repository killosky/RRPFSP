import torch
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_normal_distribution(a):
    # 计算标准差（方差的平方根）
    std_dev = np.sqrt(a / 20)

    # 生成数据
    data = np.random.normal(a, std_dev, 10000)

    # 绘制分布图
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=40, color='blue', alpha=0.7)
    plt.title(f'Normal Distribution with Mean = {a} and Variance = {a/20}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # plot_normal_distribution(20)  # 举例，这里可以替换为任何正值
    # a = torch.tensor([1, 0, 0])
    # print(torch.nonzero(a==0))
    # print(a)


    # # 定义矩阵的大小
    # n = 5  # 例如，创建一个5x5的矩阵
    #
    # # 使用torch.diag创建一个对角线上的元素为1的矩阵
    # # 第二个参数为1，意味着填充在主对角线之上的第一个对角线上
    # matrix = torch.diag(torch.ones(n - 1), 1).long()
    #
    # print(matrix)
    a = torch.ones(size=(3, 4, 5))
    print(torch.mean(a, dim=(-1, -2)).size())
