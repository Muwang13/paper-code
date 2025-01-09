import argparse
import random
import time
import numpy as np
import pandas as pd
import torch

from copy import deepcopy
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader
from model import BasicCNN as Model, CNN_mnist as CNN
from model import weight_init
from dataset import dirichlet_split, plot_label_distribution, get_client_alpha, get_client_beta, dirichlet_data


class FedSystem(object):
    def __init__(self, args):
        self.args = args
        self.server_model = CNN().to(args.device)
        self.client_model_set = [CNN() for _ in range(args.n_client)]
        self.server_omega = dict()
        self.client_omega_set = [dict() for _ in range(args.n_client)]
        # self.train_set_group, self.test_set = dirichlet_split(data_name=args.data, num_users=args.n_client, alpha=args.alpha, num_samples_per_client=1000)
        self.train_set_group, self.test_set = dirichlet_data(data_name=args.data, num_users=args.n_client, alpha=args.alpha)
        # 绘制客户端的数据标签分布
        # plot_label_distribution(args.n_client, self.train_set_group)
        print("客户端数据量：", [len(ts.idxs) for ts in self.train_set_group])
        self.train_loader_group = [DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True) for train_set in self.train_set_group]
        self.test_loader = DataLoader(self.test_set, batch_size=args.test_batch_size, shuffle=True)
        # 设置模型聚合权重
        if args.weight == 1:
            print('基于数据量设置聚合权重')
            self.client_alpha = get_client_alpha(self.train_set_group)
        elif args.weight == 0:
            print('基于数据量和信息熵设置聚合权重')
            self.client_alpha = get_client_beta(self.train_set_group)
        print("聚合权重：", self.client_alpha)
        self.criterion = nn.CrossEntropyLoss()

    def server_excute(self):
        start = 0
        device = self.args.device
        args = self.args

        # 初始化模型参数
        self.server_model.apply(weight_init)
        init_state_dict = self.server_model.state_dict()
        for client_idx in range(args.n_client):
            self.client_model_set[client_idx].load_state_dict(init_state_dict)
            for name, param in deepcopy(self.client_model_set[client_idx]).named_parameters():
                self.client_omega_set[client_idx][name] = torch.zeros_like(param.data).to(device)
        for name, param in deepcopy(self.server_model).named_parameters():
            self.server_omega[name] = torch.zeros_like(param.data).to(device)

        client_idx_list = [i for i in range(args.n_client)]
        activate_client_num = int(args.activate_rate * args.n_client)
        assert activate_client_num > 1
        # 训练模型
        acc_list, loss_list = [],[]
        max_acc = 0
        lr = args.lr
        lr_decay = args.decay
        for r in range(start, args.n_round):
            start_time = time.time()
            round_num = r + 1
            print('round_num -- ', round_num)

            if args.activate_rate<1:
                activate_clients = random.sample(client_idx_list, activate_client_num)
            else:
                activate_clients = client_idx_list
            alpha_sum = sum([self.client_alpha[idx] for idx in activate_clients])
            local_loss = []
            for client_idx in activate_clients:
                loss = self.client_update(client_idx, round_num, lr)
                local_loss.append(loss)
            loss_list.append(sum(local_loss) / len(local_loss))

            new_param = {}
            new_omega = {}
            with torch.no_grad():
                for name, param in self.server_model.named_parameters():
                    new_param[name] = param.data.zero_()
                    new_omega[name] = self.server_omega[name].data.zero_()
                    for client_idx in activate_clients:
                        new_param[name] += (self.client_alpha[client_idx]/alpha_sum) * self.client_omega_set[client_idx][name] * \
                                           self.client_model_set[client_idx].state_dict()[name].to(device)
                        new_omega[name] += (self.client_alpha[client_idx]/alpha_sum) * self.client_omega_set[client_idx][name]
                    new_param[name] /= (new_omega[name] + args.eps)

                # for name, param in self.server_model.named_parameters():
                    self.server_model.state_dict()[name].data.copy_(new_param[name])  # https://discuss.pytorch.org/t/how-can-i-modify-certain-layers-weight-and-bias/11638
                    self.server_omega[name] = new_omega[name]
                    for client_idx in range(args.n_client):
                        self.client_model_set[client_idx].state_dict()[name].data.copy_(new_param[name].cpu())
                        self.client_omega_set[client_idx][name].data.copy_(new_omega[name])

            acc = self.test_server_model()
            max_acc = max(max_acc, acc)
            acc_list.append(acc)
            print(f'******* round = {r + 1} | acc = {round(acc, 4)} | max_acc = {round(max_acc, 4)} *******')

            lr = lr * lr_decay
            # 绘制图像
            if round_num % 300 == 0:
                # 准确率图像
                plt.figure()
                plt.plot(range(len(acc_list)), acc_list)
                plt.ylabel('Accuracy')
                plt.xlabel('epoch')
                plt.savefig('results/ours/figures/acc/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_decay_{}_bs_{}_w_{}_seed_{}.png'.format(
                        args.n_client, args.data, args.activate_rate, args.alpha, round_num, args.lr, args.decay,
                        args.n_epoch, args.weight, args.i_seed))

                # 损失图像
                plt.figure()
                plt.plot(range(len(loss_list)), loss_list)
                plt.ylabel('Loss')
                plt.xlabel('epoch')
                plt.savefig('results/ours/figures/loss/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_decay_{}_bs_{}_w_{}_seed_{}.png'.format(
                        args.n_client, args.data, args.activate_rate, args.alpha, round_num, args.lr, args.decay,
                        args.n_epoch, args.weight, args.i_seed))

                # 保存准确率数据
                df1 = pd.DataFrame(acc_list, columns=['accuracy'])
                df1.to_excel('results/ours/data/acc/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_decay_{}_bs_{}_w_{}_seed_{}.xlsx'.format(
                        args.n_client, args.data, args.activate_rate, args.alpha, round_num, args.lr, args.decay,
                        args.n_epoch, args.weight, args.i_seed))

                # 保存损失数据
                df2 = pd.DataFrame(loss_list, columns=['loss'])
                df2.to_excel('results/ours/data/loss/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_decay_{}_bs_{}_w_{}_seed_{}.xlsx'.format(
                        args.n_client, args.data, args.activate_rate, args.alpha, round_num, args.lr, args.decay,
                        args.n_epoch, args.weight, args.i_seed))

                # 保存模型
                torch.save(self.server_model,'results/ours/models/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_decay_{}_bs_{}_w_{}_seed_{}.pt'.format(
                               args.n_client, args.data, args.activate_rate, args.alpha, round_num, args.lr, args.decay,
                               args.n_epoch, args.weight, args.i_seed))

                print("save data successfully!")
            end_time = time.time()
            print('---epoch time: %s seconds ---' % round((end_time - start_time), 2))
        # return acc_list

    def get_csd_loss(self, client_idx, mu, omega, round_num):
        loss_set = []
        for name, param in self.client_model_set[client_idx].named_parameters():
            theta = self.client_model_set[client_idx].state_dict()[name]
            # omega_dropout = torch.rand(omega[name].size()).cuda() if cuda else torch.rand(omega[name].size())
            # omega_dropout[omega_dropout>0.5] = 1.0
            # omega_dropout[omega_dropout <= 0.5] = 0.0

            loss_set.append((0.5 / round_num) * (omega[name] * ((theta - mu[name]) ** 2)).sum())

        return sum(loss_set)

    def client_update(self, client_idx, round_num, lr):
        log_ce_loss = 0
        log_csd_loss = 0
        device = self.args.device
        self.client_model_set[client_idx] = self.client_model_set[client_idx].to(device)
        optimizer = optim.SGD(self.client_model_set[client_idx].parameters(), lr=lr)

        new_omega = dict()
        new_mu = dict()
        server_model_state_dict = self.server_model.state_dict()
        for name, param in self.client_model_set[client_idx].named_parameters():
            new_omega[name] = deepcopy(self.server_omega[name])
            new_mu[name] = deepcopy(server_model_state_dict[name])

        self.client_model_set[client_idx].train()
        for epoch in range(args.n_epoch):
            for batch_idx, (data, target) in enumerate(self.train_loader_group[client_idx]):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.client_model_set[client_idx](data)
                ce_loss = self.criterion(output, target)
                csd_loss = self.get_csd_loss(client_idx, new_mu, new_omega, round_num) if args.csd_importance > 0 else 0
                ce_loss.backward(retain_graph=True)

                for name, param in self.client_model_set[client_idx].named_parameters():
                    if param.grad is not None:
                        self.client_omega_set[client_idx][name] += (len(target) / len(
                            self.train_set_group[client_idx])) * param.grad.data.clone() ** 2

                optimizer.zero_grad()
                loss = ce_loss + args.csd_importance * csd_loss
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.client_model_set[client_idx].parameters(), args.clip)
                optimizer.step()
                log_ce_loss += ce_loss.item()
                log_csd_loss += csd_loss.item() if args.csd_importance > 0 else 0

        log_ce_loss /= args.n_epoch
        log_csd_loss /= (args.n_epoch / args.csd_importance) if args.csd_importance > 0 else 1
        print(f'client_idx = {client_idx + 1} | loss = {log_ce_loss + log_csd_loss} (ce: {log_ce_loss} + csd: {log_csd_loss})')
        self.client_model_set[client_idx] = self.client_model_set[client_idx].cpu()

        return log_ce_loss + log_csd_loss

    def test_server_model(self):
        self.server_model.eval()
        correct = 0
        n_test = 0
        device = self.args.device
        for data, target in self.test_loader:
            data, target = data.to(device), target.to(device)
            scores = self.server_model(data)
            _, predicted = scores.max(1)
            correct += predicted.eq(target.view_as(predicted)).sum().item()
            n_test += data.size(0)
        return correct / n_test

if __name__ == '__main__':
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mnist')                                                         # 数据集
    parser.add_argument('--n_round', type=int, default=300)                                                          # 联邦学习轮数
    parser.add_argument('--n_client', type=int, default=20)                                                          # 客户端数量
    parser.add_argument('--activate_rate', type=float, default=0.5)                                                  # 激活客户端比例
    parser.add_argument('--n_epoch', type=int, default=1)                                                            # 客户端训练轮数
    parser.add_argument('--lr', type=float, default=0.1)                                                             # 学习率
    parser.add_argument('--alpha', type=float, default=0.01)                                                         # Dirichlet分布参数
    parser.add_argument('--decay', type=float, default=1)                                                            # 学习率衰减
    parser.add_argument('--pruing_p', type=float, default=0)                                                         # 剪枝比例
    parser.add_argument('--csd_importance', type=float, default=0)                                                   # 控制变量损失权重
    parser.add_argument('--eps', type=float, default=1e-5)                                                           # 表征一个极小数，避免除0
    parser.add_argument('--clip', type=float, default=10)                                                            # 梯度裁剪
    parser.add_argument('--train_batch_size', type=int, default=32)                                                  # 客户端训练批次大小
    parser.add_argument('--test_batch_size', type=int, default=64)                                                   # 测试批次大小
    parser.add_argument('--i_seed', type=int, default=20001)                                                         # 随机种子
    parser.add_argument('--weight', type=int, default=0, help='1: datasize, 0: datasize_entropy')                    # 聚合权重，
    args = parser.parse_args()

    # 训练设备
    cuda = torch.cuda.is_available()
    args.device = torch.device('cuda') if cuda else torch.device('cpu')
    print('device: ', args.device)

    print(f"data {args.data}, n_round {args.n_round}, n_client {args.n_client}, activate rate {args.activate_rate}, n_epoch {args.n_epoch}, "
          f"lr {args.lr}, alpha {args.alpha}, batch_size {args.train_batch_size}, pruing_p {args.pruing_p}, "
          f"csd_importance {args.csd_importance}, seed {args.i_seed}, decay {args.decay}")

    # 设置随机数种子
    np.random.seed(args.i_seed)
    torch.manual_seed(args.i_seed)
    random.seed(args.i_seed)

    fed_sys = FedSystem(args)
    fed_sys.server_excute()