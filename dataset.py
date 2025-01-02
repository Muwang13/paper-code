import numpy as np
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

    # 构建每个客户端的子集数据
    # client_datasets = []
    # for client_id in range(num_users):
    #     client_datasets.append(Subset(dataset, client_data[client_id]))

    # return client_datasets, dirichlet_weights, client_data, dataset
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

if __name__ == '__main__':
    # 设置参数
    num_users = 10  # 假设有10个客户端
    alpha = 0.5  # Dirichlet分布的alpha参数
    num_samples_per_client = 1000  # 每个客户端有1000个样本
    data_name = 'cifar10'  # 使用CIFAR-10数据集

    # 使用Dirichlet分布划分数据
    train_set_group, test_set = dirichlet_split(data_name, num_users, alpha, num_samples_per_client)
    plot_label_distribution(num_users, train_set_group)
