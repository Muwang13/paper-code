import pandas as pd


def save_fig():
    pass


def save_data(args, acc_list, loss_train, loss_test):
    df1 = pd.DataFrame(acc_list, columns=['accuracy'])
    df1.to_excel(args.root + 'data/acc/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_csd_{}_decay_{}_bs_{}_w_{}_seed_{}.xlsx'.format(
            args.n_client, args.data, args.activate_rate, args.alpha, args.n_round, args.lr, args.csd_importance,
            args.decay, args.n_epoch, args.weight, args.i_seed))

    # 保存损失数据
    df2 = pd.DataFrame(loss_train, columns=['train loss'])
    df2.to_excel(args.root + 'data/train_loss/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_csd_{}_decay_{}_bs_{}_w_{}_seed_{}.xlsx'.format(
            args.n_client, args.data, args.activate_rate, args.alpha, args.n_round, args.lr, args.csd_importance,
            args.decay, args.n_epoch, args.weight, args.i_seed))

    df3 = pd.DataFrame(loss_test, columns=['test loss'])
    df3.to_excel(args.root + 'data/test_loss/users_{}_data_{}_C{}_alpha_{}_round_{}_lr_{}_csd_{}_decay_{}_bs_{}_w_{}_seed_{}.xlsx'.format(
            args.n_client, args.data, args.activate_rate, args.alpha, args.n_round, args.lr, args.csd_importance,
            args.decay, args.n_epoch, args.weight, args.i_seed))