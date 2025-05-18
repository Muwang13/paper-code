import pandas as pd
from matplotlib import pyplot as plt

def save_fig(args, acc_list, loss_list, round_num):
    # 准确率图像
    plt.figure()
    plt.plot(acc_list)
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.title('Test Accuracy')
    plt.savefig(args.root + 'figures/acc/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_csd_{}_decay_{}_bs_{}_w_{}_seed_{}.png'.format(
            args.n_client, args.data, args.activate_rate, args.alpha, round_num, args.lr, args.csd_importance,
            args.decay, args.n_epoch, args.weight, args.i_seed))
    plt.close()

    plt.figure()
    plt.plot(loss_list)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.title('Test Loss')
    plt.savefig(args.root + 'figures/loss/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_csd_{}_decay_{}_bs_{}_w_{}_seed_{}.png'.format(
            args.n_client, args.data, args.activate_rate, args.alpha, round_num, args.lr, args.csd_importance,
            args.decay, args.n_epoch, args.weight, args.i_seed))
    plt.close()


def save_data(args, acc_list, loss_list):
    df1 = pd.DataFrame(acc_list, columns=['accuracy'])
    df1.to_excel(args.root + 'data/acc/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_csd_{}_decay_{}_bs_{}_w_{}_seed_{}.xlsx'.format(
            args.n_client, args.data, args.activate_rate, args.alpha, args.n_round, args.lr, args.csd_importance,
            args.decay, args.n_epoch, args.weight, args.i_seed))

    df2 = pd.DataFrame(loss_list, columns=['test loss'])
    df2.to_excel(args.root + 'data/loss/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_csd_{}_decay_{}_bs_{}_w_{}_seed_{}.xlsx'.format(
            args.n_client, args.data, args.activate_rate, args.alpha, args.n_round, args.lr, args.csd_importance,
            args.decay, args.n_epoch, args.weight, args.i_seed))

    print("Save successfully!")
