import random
import warnings

import numpy as np
import math
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from copy import deepcopy
from cnn_lab.autoaugment import CIFAR10Policy, Cutout

# 忽略 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 设置随机种子以确保可重复性
np.random.seed(41)
torch.manual_seed(41)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.indices = [int(i) for i in idxs]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        img, label = self.dataset[self.indices[item]]
        return img, label

def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return trainset, testset

def load_fmnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

# 生成Dirichlet分布划分数据: 1. 客户端数据量相同（1000）
def dirichlet_iid(data_name, num_users=10, alpha=100, num_samples_per_client=1000):
    if data_name == 'fmnist':
        dataset, test_dataset = load_fmnist()
    elif data_name == 'cifar10':
        dataset, test_dataset = load_cifar10()
    else:
        print ('Data name error')
        return None

    num_classes = len(test_dataset.classes)
    targets = np.array(dataset.targets)

    # 创建空字典，用来存放每个客户端的数据索引
    client_data = {i: [] for i in range(num_users)}
    # 对每个客户端生成Dirichlet分布的权重
    dirichlet_weights = np.random.dirichlet(np.repeat(alpha, num_classes), num_users)
    # 对每个客户端，根据Dirichlet分布的权重划分样本
    for client_id in range(num_users):
        # 创建一个长度为num_classes的列表，存放每个类别的样本数
        class_counts = (dirichlet_weights[client_id] * num_samples_per_client).astype(int)

        # 对每个类别进行样本分配
        for class_id in range(num_classes):
            # 找出该类别的所有样本索引
            class_indices = np.where(targets == class_id)[0]
            # 随机打乱类别内样本的顺序
            np.random.shuffle(class_indices)
            # 按照class_counts的要求，选择样本
            selected_indices = list(class_indices[:class_counts[class_id]])
            # 将选中的样本添加到对应客户端的样本列表中
            client_data[client_id].extend(selected_indices)

    return [DatasetSplit(deepcopy(dataset), client_data[i]) for i in range(num_users)], test_dataset

# 生成Dirichlet分布划分数据: 2. 客户端数据量不同
def dirichlet_non(data_name, num_users=10, alpha = 100):
    if data_name == 'fmnist':
        dataset, test_dataset = load_fmnist()
    elif data_name == 'cifar10':
        dataset, test_dataset = load_cifar10()
    else:
        print ('Data name error')
        return None

    class_num = 10
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(len(dataset.targets))
    labels = np.asarray(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    class_lableidx = [idxs_labels[:, idxs_labels[1, :] == i][0, :] for i in range(class_num)]

    sample_matrix = np.random.dirichlet([alpha for _ in range(num_users)], class_num).T
    class_sampe_start = [0 for i in range(class_num)]

    def sample_rand(rand, class_sampe_start):
        class_sampe_end = [start + int(len(class_lableidx[sidx]) * rand[sidx]) for sidx, start in enumerate(class_sampe_start)]
        rand_set = np.array([])
        for eidx, rand_end in enumerate(class_sampe_end):
            rand_start = class_sampe_start[eidx]
            if rand_end<= len(class_lableidx[eidx]):
                rand_set = np.concatenate([rand_set, class_lableidx[eidx][rand_start:rand_end]], axis=0)

            else:
                if rand_start< len(class_lableidx[eidx]):
                    rand_set = np.concatenate([rand_set, class_lableidx[eidx][rand_start:]],axis=0)
                else:
                    rand_set=np.concatenate([rand_set,random.sample(class_lableidx[eidx] , rand_end - rand_start +1)],axis=0)
        if rand_set.shape[0] == 0:
            rand_set = np.concatenate([rand_set, class_lableidx[0][0:1]], axis=0)
        return rand_set, class_sampe_end

    for i in range(num_users):
        rand_set, class_sampe_start = sample_rand(sample_matrix[i], class_sampe_start)
        dict_users[i] = rand_set

    return [DatasetSplit(deepcopy(dataset), dict_users[i]) for i in range(num_users)], test_dataset

# 生成Dirichlet分布划分数据: 3. 客户端数据量（特定的划分方式）
def dirichlet_fixed(data_name, num_users=50, alpha=0.1, fraction=0.7):
    if data_name == 'cifar10':
        trainset, testset = load_cifar10()
    elif data_name == 'fmnist':
        trainset, testset = load_fmnist()

    num_clients_dirichlet = num_users // 2
    num_clients_iid = num_users - num_clients_dirichlet

    # 获取训练集的标签
    labels = np.array(trainset.targets)

    # 将数据集按类别分类
    num_classes = len(testset.classes)
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    total_samples = len(trainset)
    iid_samples_per_client = int(total_samples*0.3) // num_classes

    # 使用IID方式分配 30%数据给25个客户端
    client_data_iid = [[] for _ in range(num_clients_iid)]
    samples_per_client_iid = int(total_samples * (1-fraction)) // num_clients_iid  # 每个客户端的数据量

    remaining_indices = []
    for i in range(num_classes):
        remaining_indices.extend(class_indices[i][:iid_samples_per_client])
        class_indices[i] = class_indices[i][iid_samples_per_client:]

    # 打乱顺序
    np.random.shuffle(remaining_indices)

    # 平均分配数据
    for client_idx in range(num_clients_iid):
        start_idx = client_idx * samples_per_client_iid
        end_idx = start_idx + samples_per_client_iid
        client_data_iid[client_idx].extend(remaining_indices[start_idx:end_idx])

    # 使用Dirichlet分布分配70%的数据给25个客户端
    client_data_dirichlet = [[] for _ in range(num_clients_dirichlet)]
    samples_per_client_dirichlet = int(total_samples * fraction) // num_clients_dirichlet  # 每个客户端的数据量

    for client_idx in range(num_clients_dirichlet):
        # 为每个客户端生成一个标签分布（基于Dirichlet分布）
        proportions = np.random.dirichlet(np.ones(num_classes) * alpha)
        proportions = proportions / proportions.sum()  # 归一化

        # 根据标签分布分配数据
        for class_idx in range(num_classes):
            class_data = class_indices[class_idx]
            np.random.shuffle(class_data)
            num_samples_class = int(proportions[class_idx] * samples_per_client_dirichlet)

            client_data_dirichlet[client_idx].extend(class_data[:num_samples_class])
            class_indices[class_idx] = class_data[num_samples_class:]

        # 如果数据量不足，随机补充
        while len(client_data_dirichlet[client_idx]) < samples_per_client_dirichlet:
            class_idx = np.random.randint(0, num_classes)
            class_data = class_indices[class_idx]
            if len(class_data) > 0:
                client_data_dirichlet[client_idx].append(class_data[0])
                class_indices[class_idx] = np.delete(class_indices[class_idx], 0)

    # 合并所有客户端的数据
    client_data = client_data_dirichlet + client_data_iid

    # 打印每个客户端的数据量和标签分布
    # for client_idx in range(num_clients):
    #     client_labels = labels[client_data[client_idx]]
    #     label_distribution = np.bincount(client_labels, minlength=num_classes)
    #     print(f"Client {client_idx} has {len(client_data[client_idx])} samples, label distribution: {label_distribution}")

    # return client_data, testset
    return [DatasetSplit(trainset, client_data[i]) for i in range(num_users)], testset


# 根据客户端数据量大小设置权重
def get_client_alpha(train_set_group):
    client_n_sample = [len(ts.indices) for ts in train_set_group]
    total_n_sample = sum(client_n_sample)
    client_alpha = [n_sample / total_n_sample for n_sample in client_n_sample]
    # print(f'alpha = {client_alpha}')
    return client_alpha

# 基于客户端数据量和信息熵设置权重
def get_client_beta(train_set_group):
    client_n_sample = [len(ts.indices) for ts in train_set_group]
    client_entropy = []
    labels_num = []
    for idx, ts in enumerate(train_set_group):
        if len(train_set_group[0].dataset.targets) == 50000: # Cifar10
            targets = [ts.dataset.targets[idx] for idx in ts.indices]
        elif len(train_set_group[0].dataset.targets) == 60000: # Fashion0-Mnist
            targets = [ts.dataset.targets[idx].item() for idx in ts.indices]
        labels = set(targets)
        # print(f"client {idx}: {labels}")
        labels_num.append(len(labels))
        counts = [targets.count(label) for label in labels]
        client_entropy.append(get_entropy(counts))

    client_beta = [i*j for i,j in zip(client_n_sample, client_entropy)]
    total = sum(client_beta)
    client_beta = [x/total for x in client_beta]
    print(f"label classes: {labels_num}")
    return client_beta

# 计算信息熵：进一步细化到[1, num]，其中 num 为标签类别总数
def get_entropy(list):
    entropy = 0
    total = sum(list)
    list = [i / total for i in list]
    for p in list:
        entropy -= p * math.log(p, 2)
    return pow(2,entropy)

# 绘制客户端的数据标签分布
def plot_label_distribution(num_users, train_set_group):
    labels = train_set_group[0].dataset.targets
    client_idx = [item.indices for item in train_set_group]
    print(f'客户端数据量：{[len(x) for x in client_idx]}')
    # 展示不同client上的label分布
    plt.figure(figsize=(12, 8))
    label_distribution = [[] for _ in range(10)]
    for c_id, idc in enumerate(client_idx):
        for idx in idc:
            label_distribution[labels[idx]].append(c_id)
    plt.hist(label_distribution, stacked=True,
             bins=np.arange(-0.5, num_users + 1.5, 1),
             label=train_set_group[0].dataset.classes, rwidth=0.5)
    plt.xticks(np.arange(num_users), ["%d" % c_id for c_id in range(num_users)])
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.title("Data Distribution of All Clients")
    plt.tight_layout()
    # plt.savefig('results/0221/figs/clients_label_distribution.png')
    plt.show()

def plot_label_samples(train_set_group, num_clients_dirichlet=25):
    num_clients = len(train_set_group)
    # 从两类客户端中各选取5个客户端
    selected_clients_dirichlet = np.random.choice(range(num_clients_dirichlet), 5, replace=False)
    selected_clients_iid = np.random.choice(range(num_clients_dirichlet, num_clients), 5, replace=False)

    # 合并选中的客户端
    selected_clients = np.concatenate([selected_clients_dirichlet, selected_clients_iid])

    # 准备绘图数据
    client_ids = [f"Client {idx}" for idx in selected_clients]
    label_distributions = []

    labels = np.array(train_set_group[0].dataset.targets)
    num_classes = len(train_set_group[0].dataset.classes)

    for client_idx in selected_clients:
        client_labels = labels[train_set_group[client_idx].indices]
        label_distribution = np.bincount(client_labels, minlength=num_classes)
        label_distributions.append(label_distribution)

    # 将标签分布转换为数组
    label_distributions = np.array(label_distributions)

    # 绘制堆叠柱状图
    plt.figure(figsize=(12, 6))
    x = np.arange(len(selected_clients))  # 横坐标：客户端ID
    bottom = np.zeros(len(selected_clients))  # 用于堆叠的初始值

    # 为每个标签绘制堆叠部分
    colors = plt.cm.tab10.colors  # 使用tab10颜色映射
    for class_idx in range(num_classes):
        plt.bar(x, label_distributions[:, class_idx], width=0.6, bottom=bottom, color=colors[class_idx], label=f"Class {class_idx}")
        bottom += label_distributions[:, class_idx]  # 更新堆叠的底部

    # 设置横坐标和标签
    plt.xticks(x, client_ids, rotation=45)
    plt.xlabel("Client ID")
    plt.ylabel("Number of Samples")
    plt.title("Data Distribution of Selected Clients")
    plt.legend(title="Class Label", bbox_to_anchor=(1.05, 1), loc='upper left')

    # 显示图形
    plt.tight_layout()
    # plt.savefig('results/0303/figs/clients_label_distribution_sample.png')
    plt.show()

if __name__ == '__main__':
    # 设置参数
    num_users = 50  # 假设有10个客户端
    alpha = 0.05  # Dirichlet分布的alpha参数
    num_samples_per_client = 1000  # 每个客户端有1000个样本
    data_name = 'cifar10'  # 使用CIFAR-10数据集

    # 使用Dirichlet分布划分数据（数据量相等）
    # train_set_group, test_set = dirichlet_iid(data_name, num_users, alpha, num_samples_per_client)

    # 使用Dirichlet分布划分数据（数据量不等）
    train_set_group, test_set = dirichlet_iid(data_name, num_users, alpha)
    plot_label_distribution(num_users, train_set_group)
