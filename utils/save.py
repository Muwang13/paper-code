import pandas as pd
from matplotlib import pyplot as plt


def save_fig(args, acc_list, loss_train, loss_test, round_num):
    # 准确率图像
    plt.figure()
    plt.plot(range(len(acc_list)), acc_list)
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.title('Test Accuracy')
    plt.savefig(args.root + 'figures/acc/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_csd_{}_decay_{}_bs_{}_w_{}_seed_{}.png'.format(
            args.n_client, args.data, args.activate_rate, args.alpha, round_num, args.lr, args.csd_importance,
            args.decay, args.n_epoch, args.weight, args.i_seed))

    # 损失图像
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.title('Train Loss')
    plt.savefig(args.root + 'figures/train_loss/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_csd_{}_decay_{}_bs_{}_w_{}_seed_{}.png'.format(
            args.n_client, args.data, args.activate_rate, args.alpha, round_num, args.lr, args.csd_importance,
            args.decay, args.n_epoch, args.weight, args.i_seed))

    plt.figure()
    plt.plot(range(len(loss_test)), loss_test)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.title('Test Loss')
    plt.savefig(args.root + 'figures/test_loss/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_csd_{}_decay_{}_bs_{}_w_{}_seed_{}.png'.format(
            args.n_client, args.data, args.activate_rate, args.alpha, round_num, args.lr, args.csd_importance,
            args.decay, args.n_epoch, args.weight, args.i_seed))


def save_data(args, acc_list, loss_train, loss_test):
    df1 = pd.DataFrame(acc_list, columns=['accuracy'])
    df1.to_excel(args.root + 'data/acc/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_csd_{}_decay_{}_bs_{}_w_{}_seed_{}.xlsx'.format(
            args.n_client, args.data, args.activate_rate, args.alpha, args.n_round, args.lr, args.csd_importance,
            args.decay, args.n_epoch, args.weight, args.i_seed))

    # 保存损失数据
    df2 = pd.DataFrame(loss_train, columns=['train test_loss'])
    df2.to_excel(args.root + 'data/train_loss/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_csd_{}_decay_{}_bs_{}_w_{}_seed_{}.xlsx'.format(
            args.n_client, args.data, args.activate_rate, args.alpha, args.n_round, args.lr, args.csd_importance,
            args.decay, args.n_epoch, args.weight, args.i_seed))

    df3 = pd.DataFrame(loss_test, columns=['test test_loss'])
    df3.to_excel(args.root + 'data/test_loss/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_csd_{}_decay_{}_bs_{}_w_{}_seed_{}.xlsx'.format(
            args.n_client, args.data, args.activate_rate, args.alpha, args.n_round, args.lr, args.csd_importance,
            args.decay, args.n_epoch, args.weight, args.i_seed))