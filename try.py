import torch
import numpy as np
import matplotlib.pyplot as plt

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
    a = torch.tensor([1, 0, 0])
    print(torch.nonzero(a==0))
    print(a)