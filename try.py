import torch
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import transpose_list_of_tensors

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
    # a = torch.ones(size=(3, 4, 5))
    # print(torch.mean(a, dim=(-1, -2)).size())
    # a1 = torch.tensor([1, 2, 3])
    # a2 = torch.tensor([4, 5])
    # a3 = torch.tensor([7, 8, 9])
    # a4 = torch.tensor([10, 11, 12, 13])
    # a5 = torch.tensor([13, 14, 15])
    # a6 = torch.tensor([16, 17])
    # list1 = [[a1, a2, a3], [a4, a5, a6]]
    # # list2 = [[a1, a4], [a2, a5], [a3, a6]]
    # print(list1)
    # # print(list2)
    # # print(transpose_list_of_tensors(list1))
    # print([item for sublist in transpose_list_of_tensors(list1) for item in sublist])
    a = torch.tensor([-10, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
    # a = []
    # a = torch.stack(a)
    # batch_idxes = torch.arange(0, 1, dtype=torch.long)
    print(torch.nonzero(a))

    # print(torch.mean(a), a.std())
    # print(a)
    # print(torch.argmax(a))

