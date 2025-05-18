import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 设置全局样式
plt.style.use('ggplot')
plt.rcParams.update({
    'font.sans-serif': ['SimHei'],
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

# 颜色、线型、标记
COLORS = {'FedAvg': '#1f77b4', 'FOLA': '#2ca02c', 'Ours': '#d62728'}
LINE_STYLES = {'with_selection': '-.', 'without_selection': '--'}
MARKERS = {'FedAvg': 'o', 'FOLA': 's', 'Ours': 'D'}


# 加载与平滑数据
def load_and_process_data(file_paths, window_sizes):
    return [
        savgol_filter(pd.read_excel(path).iloc[:, 1].values, window, 1)
        for path, window in zip(file_paths, window_sizes)
    ]


# 绘图函数
def plot_all_methods(data, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    method_names = ['FedAvg', 'FOLA', 'Ours']

    for i, method in enumerate(method_names):
        with_sel = data[i * 2]
        without_sel = data[i * 2 + 1]

        ax.plot(with_sel, label=f'{method} (Datasize)', linewidth=2,
                linestyle=LINE_STYLES['with_selection'], color=COLORS[method],
                marker=MARKERS[method], markevery=30, markersize=6)

        ax.plot(without_sel, label=f'{method} (Data Prior)', linewidth=2,
                linestyle=LINE_STYLES['without_selection'], color=COLORS[method],
                marker=MARKERS[method], markevery=30, markersize=6)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10, framealpha=1, shadow=True, ncol=2)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', transparent=False, facecolor='white')
    plt.show()


# 文件路径
cifar10_files = [
    'results/0221/fedavg/data/acc/users_50_data_cifar10_C0.2_alpha_0.01_round_300_lr_0.01_csd_1_decay_1_bs_5_w_1_seed_10001.xlsx',
    'results/0221/fedavg/data/acc/users_50_data_cifar10_C0.2_alpha_0.01_round_300_lr_0.01_csd_1_decay_1_bs_5_w_0_seed_10001.xlsx',
    'results/0221/fola/data/acc/users_50_data_cifar10_C0.2_alpha_0.01_round_300_lr_0.01_csd_1_decay_1_bs_5_w_1_seed_10001.xlsx',
    'results/0221/fola/data/acc/users_50_data_cifar10_C0.2_alpha_0.01_round_300_lr_0.01_csd_1_decay_1_bs_5_w_0_seed_10001.xlsx',
    'results/0221/ours/data/acc/users_50_data_cifar10_C0.2_alpha_0.01_round_300_lr_0.01_csd_1_decay_1_bs_5_w_1_seed_10001_new.xlsx',
    'results/0221/ours/data/acc/users_50_data_cifar10_C0.2_alpha_0.01_round_300_lr_0.01_csd_1_decay_1_bs_5_w_0_seed_10001_new.xlsx'
]

fmnist_files = [
    'results/0221/fedavg/data/acc/users_50_data_fmnist_C0.2_alpha_0.01_round_300_lr_0.01_csd_1_decay_1_bs_1_w_1_seed_10001.xlsx',
    'results/0221/fedavg/data/acc/users_50_data_fmnist_C0.2_alpha_0.01_round_300_lr_0.01_csd_1_decay_1_bs_1_w_0_seed_10001.xlsx',
    'results/0221/fola/data/acc/users_50_data_fmnist_C0.2_alpha_0.01_round_300_lr_0.01_csd_1_decay_1_bs_1_w_1_seed_10001.xlsx',
    'results/0221/fola/data/acc/users_50_data_fmnist_C0.2_alpha_0.01_round_300_lr_0.01_csd_1_decay_1_bs_1_w_0_seed_10001.xlsx',
    'results/0221/ours/data/acc/users_50_data_fmnist_C0.2_alpha_0.01_round_300_lr_0.01_csd_1_decay_1_bs_1_w_1_seed_10001.xlsx',
    'results/0221/ours/data/acc/users_50_data_fmnist_C0.2_alpha_0.01_round_300_lr_0.01_csd_1_decay_1_bs_1_w_0_seed_10001.xlsx'
]

# 加载数据
cifar10_data = load_and_process_data(cifar10_files, [41] * 6)
fmnist_data = load_and_process_data(fmnist_files, [31, 31, 31, 31, 11, 11])

# 绘图
plot_all_methods(cifar10_data, 'CIFAR-10 (α=0.01)', 'cifar10_all_results.png')
plot_all_methods(fmnist_data, 'Fashion-MNIST (α=0.01)', 'fmnist_all_results.png')
