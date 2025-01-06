import pandas as pd
import matplotlib.pyplot as plt

# 读取 Excel 文件
df1 = pd.read_excel('results/fedavg/data/acc/users_50_data_fashion_mnist_C0.2_alpha_0.1_round_300_seed_30001.xlsx')
df2 = pd.read_excel('results/fedavg/data/acc/users_50_data_fashion_mnist_C0.2_alpha_0.1_round_300_seed_30002.xlsx')
df3 = pd.read_excel('results/fedavg/data/acc/users_50_data_fashion_mnist_C0.2_alpha_0.1_round_300_seed_30003.xlsx')
df4 = pd.read_excel('results/fedavg/data/acc/users_50_data_fashion_mnist_C0.2_alpha_0.1_round_300_seed_30004.xlsx')
df5 = pd.read_excel('results/fedavg/data/acc/users_50_data_fashion_mnist_C0.2_alpha_0.1_round_300_seed_30005.xlsx')

df6 = pd.read_excel('results/fola/data/acc/users_50_data_fashion_mnist_C0.2_alpha_0.1_round_300_seed_30001.xlsx')
df7 = pd.read_excel('results/fola/data/acc/users_50_data_fashion_mnist_C0.2_alpha_0.1_round_300_seed_30002.xlsx')
df8 = pd.read_excel('results/fola/data/acc/users_50_data_fashion_mnist_C0.2_alpha_0.1_round_300_seed_30003.xlsx')
df9 = pd.read_excel('results/fola/data/acc/users_50_data_fashion_mnist_C0.2_alpha_0.1_round_300_seed_30004.xlsx')
df10 = pd.read_excel('results/fola/data/acc/users_50_data_fashion_mnist_C0.2_alpha_0.1_round_300_seed_30005.xlsx')

df11 = pd.read_excel('results/fola/data/acc/beta_users_50_data_fashion_mnist_C0.2_alpha_0.1_round_300_seed_30001.xlsx')
df12 = pd.read_excel('results/fola/data/acc/beta_users_50_data_fashion_mnist_C0.2_alpha_0.1_round_300_seed_30002.xlsx')
df13 = pd.read_excel('results/fola/data/acc/beta_users_50_data_fashion_mnist_C0.2_alpha_0.1_round_300_seed_30003.xlsx')
df14 = pd.read_excel('results/fola/data/acc/beta_users_50_data_fashion_mnist_C0.2_alpha_0.1_round_300_seed_30004.xlsx')
df15 = pd.read_excel('results/fola/data/acc/beta_users_50_data_fashion_mnist_C0.2_alpha_0.1_round_300_seed_30005.xlsx')

# 提取第一列数据（假设第一列是你要绘制的数据）
data1 = df1.iloc[:, 1]  # 从第二行开始提取数据，跳过标题行
data2 = df2.iloc[:, 1]
data3 = df3.iloc[:, 1]
data4 = df4.iloc[:, 1]
data5 = df5.iloc[:, 1]

data6 = df6.iloc[:, 1]  # 从第二行开始提取数据，跳过标题行
data7 = df7.iloc[:, 1]
data8 = df8.iloc[:, 1]
data9 = df9.iloc[:, 1]
data10 = df10.iloc[:, 1]

data11 = df11.iloc[:, 1]  # 从第二行开始提取数据，跳过标题行
data12 = df12.iloc[:, 1]
data13 = df13.iloc[:, 1]
data14 = df14.iloc[:, 1]
data15 = df15.iloc[:, 1]


fedavg_data = pd.concat([data1, data2, data3, data4, data5], axis=1)
fola_data = pd.concat([data6, data7, data8, data9, data10], axis=1)
fola_prior_data = pd.concat([data11, data12, data13, data14, data15], axis=1)

fedavg = fedavg_data.mean(axis=1)
fola = fola_data.mean(axis=1)
fola_prior = fola_prior_data.mean(axis=1)

# 绘制曲线
plt.plot(fedavg, label='fedavg')
plt.plot(fola, label='fola')
plt.plot(fola_prior, label='fola_prior')
# plt.plot(data4, label='fedavg1')
# plt.plot(data5, label='fedavg_pro')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.title('Learning curve')
plt.legend()
plt.savefig('results/plot/compare1.png')
plt.show()


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