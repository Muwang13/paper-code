import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# mnist
# 读取fedavg数据
df11 = pd.read_excel('results/fedavg/data/test_loss/users_20_data_cifar10_C0.5_alpha_0.1_round_500_lr_0.1_csd_1_decay_1_bs_1_w_1_seed_10001.xlsx')

# 读取fola数据
df21 = pd.read_excel('results/fola/data/test_loss/users_20_data_cifar10_C0.5_alpha_0.1_round_500_lr_0.1_csd_1_decay_1_bs_1_w_1_seed_10001.xlsx')

df31 = pd.read_excel('results/fola_prior/data/test_loss/users_20_data_cifar10_C0.5_alpha_0.1_round_500_lr_0.1_csd_1_decay_1_bs_1_w_1_seed_10001.xlsx')


df11_arr = savgol_filter(df11.iloc[:, 1].values, 11, 3)

df21_arr = savgol_filter(df21.iloc[:, 1].values, 11, 3)

df31_arr = savgol_filter(df31.iloc[:, 1].values, 11, 3)

plt.figure()
plt.plot(df11_arr, label='fedavg alpha 0.01', linestyle='-', c='red')

plt.plot(df21_arr, label='fola alpha 0.01', linestyle='-', c='blue')

plt.plot(df31_arr, label='ours alpha 0.01', linestyle='-', c='green')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.title('Fashion MNIST Learning curve alpha=0.01')
plt.legend()
# plt.savefig('results/plot/fmnist_alpha0.01_compare.png')
plt.show()

# 绘制曲线
# plt.plot(fedavg, label='fedavg')
# plt.plot(fola, label='fola')
# plt.plot(fola_prior, label='fola_prior')
# # plt.plot(data4, label='fedavg1')
# # plt.plot(data5, label='fedavg_pro')
# plt.xlabel('epoch')
# plt.ylabel('Accuracy')
# plt.title('Learning curve')
# plt.legend()
# plt.savefig('results/plot/compare1.png')
# plt.show()


# size = 6
# plt.figure(figsize=(10,6))
# plt.plot(FA_acc_10_90, c='red',linestyle='-',marker='o', markersize=size)
# plt.plot(FS_acc_10_90, c='blue',linestyle='-',marker='o', markersize=size)
# plt.plot(LC_acc_10_90, c='green',linestyle='-',marker='o', markersize=size)
#
# plt.plot(FA_acc_40_60, c='red',linestyle='-',marker='^', markersize=size)
# plt.plot(FS_acc_40_60, c='blue',linestyle='-',marker='^', markersize=size)
# plt.plot(LC_acc_40_60, c='green',linestyle='-',marker='^', markersize=size)
#
# plt.plot(FA_acc_70_30, c='red',marker='s', markersize=size)
# plt.plot(FS_acc_70_30, c='blue',marker='s', markersize=size)
# plt.plot(LC_acc_70_30, c='green',marker='s', markersize=size)
#
# plt.plot(FA_acc_10_90, label='FedAvg', c='red')
# plt.plot(FS_acc_10_90, label='FedSGD',c='blue')
# plt.plot(LC_acc_10_90, label='LabelClustering', c='green')
#
# plt.xlabel('Global Epoch', fontsize=15)
# plt.ylabel('Accuracy', fontsize=15)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)
# plt.grid(True)
# plt.legend(fontsize=17)
# plt.show()