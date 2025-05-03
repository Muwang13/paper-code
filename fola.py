import random
import time
import torch

from copy import deepcopy
from torch import optim, nn
from torch.utils.data import DataLoader
from model import weight_init, CIFAR10Net, FashionMNISTNet
from datas import dirichlet_non, dirichlet_iid, dirichlet_fixed
from datas import get_client_alpha, get_client_beta
from datas import plot_label_samples, plot_label_distribution
from utils.options import args_parser
from utils.save import save_data, save_fig
from utils.set_seed import set_seed


class FedSystem(object):
    def __init__(self, args):
        self.args = args
        # 如果是 mnist 或者 fmnist 数据集，使用CNN模型
        if args.data == 'mnist' or args.data == 'fmnist':
            self.server_model = FashionMNISTNet().to(args.device)
            self.client_model_set = [FashionMNISTNet() for _ in range(args.n_client)]
        else:
            # 否则使用BasicCNN模型
            self.server_model = CIFAR10Net().to(args.device)
            self.client_model_set = [CIFAR10Net() for _ in range(args.n_client)]

        print(self.server_model)
        self.server_omega = dict()
        self.client_omega_set = [dict() for _ in range(args.n_client)]

        if args.data_split == 0:
            # Dirichlet分布划分，客户端数据量相同
            self.train_set_group, self.test_set = dirichlet_iid(args.data, num_users=args.n_client, alpha=args.alpha,
                                                                num_samples_per_client=args.data_nums)
        elif args.data_split == 1:
            # 按照自己设计的数据据划分方式划分数据集
            self.train_set_group, self.test_set = dirichlet_fixed(args.data, num_users=args.n_client, alpha=args.alpha,
                                                                  fraction=args.data_fraction)
        else:
            # Dirichlet分布划分，客户端数据量不同
            self.train_set_group, self.test_set = dirichlet_non(data_name=args.data, num_users=args.n_client,
                                                                alpha=args.alpha)

        # 绘制客户端的数据标签分布
        if args.label_verbose:
            plot_label_distribution(args.n_client, self.train_set_group)

        print("客户端数据量：", [len(ts.indices) for ts in self.train_set_group])

        self.train_loader_group = [DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True) for train_set
                                   in self.train_set_group]
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
        # self.server_model.apply(weight_init)
        init_state_dict = self.server_model.state_dict()
        for client_idx in range(args.n_client):
            self.client_model_set[client_idx].load_state_dict(init_state_dict)
            for name, param in deepcopy(self.client_model_set[client_idx]).named_parameters():
                self.client_omega_set[client_idx][name] = torch.zeros_like(param.data).to(device)
        for name, param in deepcopy(self.server_model).named_parameters():
            self.server_omega[name] = torch.zeros_like(param.data).to(device)

        # 激活客户端
        client_idx_list = [i for i in range(args.n_client)]
        activate_client_num = int(args.activate_rate * args.n_client)
        assert activate_client_num > 1

        # 训练模型
        acc_list, loss_list = [], []
        max_acc = 0
        for r in range(start, args.n_round):
            start_time = time.time()
            round_num = r + 1
            print('round_num -- ', round_num)

            if args.activate_rate < 1:
                activate_clients = random.sample(client_idx_list, activate_client_num)
            else:
                activate_clients = client_idx_list
            alpha_sum = sum([self.client_alpha[idx] for idx in activate_clients])
            for client_idx in activate_clients:
                self.client_update(client_idx, round_num, args.lr)

            new_param, new_omega = {}, {}
            with torch.no_grad():
                for name, param in self.server_model.named_parameters():
                    new_param[name] = param.data.zero_()
                    new_omega[name] = self.server_omega[name].data.zero_()
                    for client_idx in activate_clients:
                        new_param[name] += (self.client_alpha[client_idx] / alpha_sum) * \
                                           self.client_omega_set[client_idx][name] * \
                                           self.client_model_set[client_idx].state_dict()[name].to(device)
                        new_omega[name] += (self.client_alpha[client_idx] / alpha_sum) * \
                                           self.client_omega_set[client_idx][name]
                    new_param[name] /= (new_omega[name] + args.eps)

                    # for name, param in self.server_model.named_parameters():
                    self.server_model.state_dict()[name].data.copy_(new_param[
                                                                        name])  # https://discuss.pytorch.org/t/how-can-i-modify-certain-layers-weight-and-bias/11638
                    self.server_omega[name] = new_omega[name]
                    for client_idx in range(args.n_client):
                        self.client_model_set[client_idx].state_dict()[name].data.copy_(new_param[name].cpu())
                        self.client_omega_set[client_idx][name].data.copy_(new_omega[name])

            acc, loss = self.test_server_model()
            max_acc = max(max_acc, acc)
            acc_list.append(acc)
            loss_list.append(loss)
            print(
                f'******* round = {r + 1} | acc = {round(acc, 4)} | max_acc = {round(max_acc, 4)} | loss = {loss} *******')

            end_time = time.time()
            print('---epoch time: %s seconds ---' % round((end_time - start_time), 2))
            # 绘制图像
            if round_num % args.plot_num == 0:
                save_fig(args, acc_list, loss_list, round_num)

        # 保存模型
        torch.save(self.server_model,
                   args.root + 'models/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_csd_{}_decay_{}_bs_{}_w_{}_seed_{}.pt'.format(
                       args.n_client, args.data, args.activate_rate, args.alpha, args.n_round, args.lr,
                       args.csd_importance, args.decay, args.n_epoch, args.weight, args.i_seed))

        return acc_list, loss_list

    def get_csd_loss(self, client_idx, mu, omega, round_num):
        loss_set = []
        for name, param in self.client_model_set[client_idx].named_parameters():
            theta = self.client_model_set[client_idx].state_dict()[name]
            loss_set.append((0.5 / round_num) * (omega[name] * ((theta - mu[name]) ** 2)).sum())

        return sum(loss_set)

    def client_update(self, client_idx, round_num, lr):
        log_ce_loss = 0
        log_csd_loss = 0
        device = self.args.device
        args = self.args
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
                ce_loss.backward(retain_graph=True)  # 第一次需要保留计算图

                for name, param in self.client_model_set[client_idx].named_parameters():
                    if param.grad is not None:
                        self.client_omega_set[client_idx][name] += (len(target) / len(
                            self.train_set_group[client_idx])) * param.grad.data.clone() ** 2

                optimizer.zero_grad()
                loss = ce_loss + args.csd_importance * csd_loss
                loss.backward()  # 第二次应该不用 retain_graph=True
                torch.nn.utils.clip_grad_norm_(self.client_model_set[client_idx].parameters(), args.clip)
                optimizer.step()
                log_ce_loss += ce_loss.item()
                log_csd_loss += csd_loss.item() if args.csd_importance > 0 else 0

        log_ce_loss /= args.n_epoch
        log_csd_loss /= (args.n_epoch / args.csd_importance) if args.csd_importance > 0 else 1
        loss = log_ce_loss + log_csd_loss
        print(f'client_idx = {client_idx + 1} | loss = {loss} (ce: {log_ce_loss} + csd: {log_csd_loss})')
        self.client_model_set[client_idx] = self.client_model_set[client_idx].cpu()

    def test_server_model(self):
        self.server_model.eval()
        correct = 0
        n_test = 0
        device = self.args.device
        test_loss = 0
        for data, target in self.test_loader:
            data, target = data.to(device), target.to(device)
            scores = self.server_model(data)
            loss = self.criterion(scores, target)
            test_loss += loss.item()
            _, predicted = scores.max(1)
            correct += predicted.eq(target.view_as(predicted)).sum().item()
            n_test += data.size(0)
        return correct / n_test, test_loss / len(self.test_loader)


def main(args):
    start_time = time.time()
    # 训练设备
    cuda = torch.cuda.is_available()
    args.device = torch.device('cuda') if cuda else torch.device('cpu')
    print('device: ', args.device)

    print(
        f"data {args.data}, n_round {args.n_round}, n_client {args.n_client}, activate rate {args.activate_rate}, n_epoch {args.n_epoch}, "
        f"lr {args.lr}, alpha {args.alpha}, batch_size {args.train_batch_size}, pruing_p {args.pruing_p}, "
        f"csd_importance {args.csd_importance}, seed {args.i_seed}, decay {args.decay}, weight {args.weight}, split {args.data_split}")

    # 设置随机数种子
    set_seed(args.i_seed)

    fed_sys = FedSystem(args)
    acc_list, loss_list = fed_sys.server_excute()

    # 保存准确率数据
    save_data(args, acc_list, loss_list)

    end_time = time.time()
    print(f'cost time: {end_time - start_time}')


if __name__ == '__main__':
    # 加载参数
    args = args_parser()
    args.n_round = 300
    args.n_client = 20
    args.activate_rate = 0.5
    args.alpha = 100
    args.i_seed = 10001
    args.split = True
    args.lr = 0.01
    args.train_batch_size = 32
    args.test_batch_size = 64
    args.data_nums = 1000

    args.label_verbose = False

    # todo 每轮修改 weight: 0: datasize_entropy, 1: datasize
    # todo FedAvg weight=1
    args.weight = 1

    # todo 每轮修改 data_split: 0: dirichlet_iid，1: dirichlet_fixed, 2: dirichlet_non
    args.data_split = 1
    # todo 每轮修改 data: fmnist, cifar10
    args.data = 'fmnist'

    # n_epoch: 客户端训练轮数(fmnist: 1, cifar10: 5)
    if args.data == 'fmnist':
        args.n_epoch = 1
    elif args.data == 'cifar10':
        args.n_epoch = 5

    args.root = f'results/0330/diri{args.data_split}/fola/'
    main(args)

# 0330 运行结果
# 1. data_split = 0 data = cifar10
# 2. data_split = 2 data = cifar10
# 3. data_split = 0 data = fmnist
# 4. data_split = 2 data = fmnist
