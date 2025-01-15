from fedavg import main as fedavg
from fola import main as fola
from fola_prior import main as fola_prior
from utils.options import args_parser

# 设置共同的参数
args = args_parser()
args.data = 'mnist'
args.n_round = 300
args.n_client = 20
args.activate_rate = 0.5
args.alpha = 0.01
args.i_seed = 10001
args.split = True
args.lr = 0.1
args.train_batch_size = 32
args.test_batch_size = 64
args.data_nums = 1000

# 参数集合
csds = [1, 10, 100]
epochs = [1, 3]

for epoch in epochs:
    for i in range(3):
        if i==0:
            print("Run AVG No CSD")
            args.root = 'results/fedavg/'
            args.n_epoch = epoch
            args.csd_importance = 0
            args.weight = 1
            fedavg(args)
        elif i==1:
            print("Run FOLA with Csd")
            args.root = 'results/fola/'
            args.n_epoch = epoch
            args.weight = 1
            for csd in csds:
                print(f"Run FOLA with Csd {csd}")
                args.csd_importance = csd
                fola(args)
        else:
            print("Run FOLA_prior with Csd")
            args.root = 'results/ours/'
            args.n_epoch = epoch
            args.weight = 0
            for csd in csds:
                print(f"Run FOLA_prior with Csd {csd}")
                args.csd_importance = csd
                fola_prior(args)



