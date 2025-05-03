import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cifar10')              # 数据集
    parser.add_argument('--n_round', type=int, default=500)                 # 联邦学习轮数
    parser.add_argument('--n_client', type=int, default=20)                 # 客户端数量
    parser.add_argument('--activate_rate', type=float, default=0.2)         # 激活客户端比例
    parser.add_argument('--n_epoch', type=int, default=5)                   # 客户端训练轮数
    parser.add_argument('--lr', type=float, default=0.1)                    # 学习率
    parser.add_argument('--alpha', type=float, default=0.1)                 # Dirichlet分布参数（越大，数据异构程度越高）
    parser.add_argument('--decay', type=float, default=1)                   # 学习率衰减
    parser.add_argument('--pruing_p', type=float, default=0)                # 剪枝比例
    parser.add_argument('--csd_importance', type=float, default=1)          # 控制变量损失权重
    parser.add_argument('--eps', type=float, default=1e-8)                  # 表征一个极小数，避免除0
    parser.add_argument('--clip', type=float, default=10)                   # 梯度裁剪
    parser.add_argument('--train_batch_size', type=int, default=32)         # 客户端训练批次大小
    parser.add_argument('--test_batch_size', type=int, default=64)          # 测试批次大小
    parser.add_argument('--i_seed', type=int, default=10001)                 # 随机种子 1000*: 数据量均匀；2000*: 数据量不均匀；3000*: 学习率衰减
    parser.add_argument('--weight', type=int, default=0, help='1: datasize, 0: datasize_entropy')                    # 聚合权重，
    # parser.add_argument('--split', type=bool, default=True, help='True：均匀划分，False：非均匀划分')  # 数据集划分方式
    parser.add_argument('--label_verbose', type=bool, default=True, help='是否绘制标签分布')  # 数据集划分方式
    parser.add_argument('--root', type=str, default='results/fedacg/')             # 结果保存路径
    parser.add_argument('--data_nums', type=int, default=1000, help='datasize for each client when splited average')
    parser.add_argument('--plot_num', type=int, default=100)
    parser.add_argument('--data_fraction', type=float, default=0.7)
    args = parser.parse_args()
    return args