import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cifar10')              # 数据集
    parser.add_argument('--n_round', type=int, default=500)                 # 联邦学习轮数
    parser.add_argument('--n_client', type=int, default=20)                 # 客户端数量
    parser.add_argument('--activate_rate', type=float, default=0.5)         # 激活客户端比例
    parser.add_argument('--n_epoch', type=int, default=1)                   # 客户端训练轮数
    parser.add_argument('--lr', type=float, default=0.1)                    # 学习率
    parser.add_argument('--alpha', type=float, default=0.1)                 # Dirichlet分布参数（越大，数据异构程度越高）
    parser.add_argument('--decay', type=float, default=1)                   # 学习率衰减
    parser.add_argument('--pruing_p', type=float, default=0)                # 剪枝比例
    parser.add_argument('--csd_importance', type=float, default=1)          # 控制变量损失权重
    parser.add_argument('--eps', type=float, default=1e-5)                  # 表征一个极小数，避免除0
    parser.add_argument('--clip', type=float, default=10)                   # 梯度裁剪
    parser.add_argument('--train_batch_size', type=int, default=32)         # 客户端训练批次大小
    parser.add_argument('--test_batch_size', type=int, default=64)          # 测试批次大小
    parser.add_argument('--i_seed', type=int, default=10001)                 # 随机种子 1000*: 数据量均匀；2000*: 数据量不均匀；3000*: 学习率衰减
    parser.add_argument('--weight', type=int, default=1, help='1: datasize, 0: datasize_entropy')                    # 聚合权重，
    parser.add_argument('--split', type=bool, default=True, help='True：均匀划分，False：非均匀划分')  # 数据集划分方式
    parser.add_argument('--label_verbose', type=bool, default=False, help='是否绘制标签分布')  # 数据集划分方式
    parser.add_argument('--root', type=str, default='results/fedacg/')             # 结果保存路径
    args = parser.parse_args()
    return args