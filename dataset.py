import random
import numpy as np
import math
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from copy import deepcopy
from cnn_lab.autoaugment import CIFAR10Policy, Cutout


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        img, label = self.dataset[self.idxs[item]]
        return img, label #self.data[item], self.targets[item]

# 生成Dirichlet分布划分数据
def dirichlet_split(data_name, num_users=10, alpha=100, num_samples_per_client=1000):
    if data_name == 'mnist':
        dataset = datasets.MNIST('./data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

        test_dataset = datasets.MNIST('./data/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    elif data_name == 'fmnist':
        dataset = datasets.FashionMNIST('./data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

        test_dataset = datasets.FashionMNIST('./data/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    elif data_name == 'cifar10':
        dataset = datasets.CIFAR10('./data/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(32, padding=4, fill=128),
                                       transforms.RandomHorizontalFlip(),
                                       CIFAR10Policy(),
                                       transforms.ToTensor(),
                                       Cutout(n_holes=1, length=16),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ]))

        test_dataset = datasets.CIFAR10('./data/', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))
    elif data_name == 'cifar100':
        dataset = datasets.CIFAR100('./data/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(32, padding=4, fill=128),
                                       transforms.RandomHorizontalFlip(),
                                       CIFAR10Policy(),
                                       transforms.ToTensor(),
                                       Cutout(n_holes=1, length=16),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ]))

        test_dataset = datasets.CIFAR100('./data/', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))
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

def plot_label_distribution(num_users, train_set_group):
    labels = train_set_group[0].dataset.targets
    client_idx = [item.idxs for item in train_set_group]
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
    plt.title("Display Label Distribution on Different Clients")
    plt.show()

def get_client_alpha(train_set_group):
    client_n_sample = [len(ts.idxs) for ts in train_set_group]
    total_n_sample = sum(client_n_sample)
    client_alpha = [n_sample / total_n_sample for n_sample in client_n_sample]
    # print(f'alpha = {client_alpha}')
    return client_alpha

def get_client_beta(train_set_group):
    client_n_sample = [len(ts.idxs) for ts in train_set_group]
    total_n_sample = sum(client_n_sample)
    client_entropy = []
    labels_num = []
    for idx, ts in enumerate(train_set_group):
        targets = [ts.dataset.targets[idx].item() for idx in ts.idxs]
        labels = set(targets)
        print(f"client {idx}: {labels}")
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

    # 可能还是需要的
    entropy = pow(2,entropy)

    return entropy

def dirichlet_data(data_name, num_users=10, alpha = 100):

    if data_name == 'mnist':
        dataset = datasets.MNIST('./data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

        test_dataset = datasets.MNIST('./data/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    elif data_name == 'fmnist':
        dataset = datasets.FashionMNIST('./data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

        test_dataset = datasets.FashionMNIST('./data/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    elif data_name == 'cifar10':
        dataset = datasets.CIFAR10('./data/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(32, padding=4, fill=128),
                                       transforms.RandomHorizontalFlip(),
                                       CIFAR10Policy(),
                                       transforms.ToTensor(),
                                       Cutout(n_holes=1, length=16),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ]))

        test_dataset = datasets.CIFAR10('./data/', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))
    elif data_name == 'cifar100':

        dataset = datasets.CIFAR100('./data/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(32, padding=4, fill=128),
                                       transforms.RandomHorizontalFlip(),
                                       CIFAR10Policy(),
                                       transforms.ToTensor(),
                                       Cutout(n_holes=1, length=16),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ]))

        test_dataset = datasets.CIFAR100('./data/', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))
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
                    rand_set=np.concatenate([rand_set, random.sample(class_lableidx[eidx] , rand_end - rand_start +1)],axis=0)
        if rand_set.shape[0] == 0:
            rand_set = np.concatenate([rand_set, class_lableidx[0][0:1]], axis=0)
        return rand_set, class_sampe_end

    for i in range(num_users):
        rand_set, class_sampe_start = sample_rand(sample_matrix[i], class_sampe_start)
        dict_users[i] = rand_set

    return [DatasetSplit(deepcopy(dataset), dict_users[i]) for i in range(num_users)], test_dataset



if __name__ == '__main__':
    # 设置参数
    num_users = 10  # 假设有10个客户端
    alpha = 0.5  # Dirichlet分布的alpha参数
    num_samples_per_client = 1000  # 每个客户端有1000个样本
    data_name = 'cifar10'  # 使用CIFAR-10数据集

    # 使用Dirichlet分布划分数据
    train_set_group, test_set = dirichlet_split(data_name, num_users, alpha, num_samples_per_client)
    plot_label_distribution(num_users, train_set_group)
