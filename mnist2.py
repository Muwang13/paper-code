import copy

from fedavg import main as fedavg
from fola import main as fola
from fola_prior import main as fola_prior
from utils.options import args_parser

# 设置共同的参数
args = args_parser()
args.data = 'mnist'
args.n_round = 300
args.n_client = 50
args.activate_rate = 0.2
args.alpha = 0.1
args.i_seed = 10001
args.split = True

args1 = copy.deepcopy(args)
args1.root = 'results/fedavg/'
args2 = copy.deepcopy(args)
args2.root = 'results/fola/'
args3 = copy.deepcopy(args)
args3.root = 'results/fola_prior/'

# 参数集合
csds = [1, 10, 100]
epochs = [1, 3]

for epoch in epochs:
    for i in range(3):
        if i==0:
            print("Run AVG No CSD")
            args1.n_epoch = epoch
            args1.csd_importance = 0
            args1.weight = 1
            fedavg(args1)
        elif i==1:
            print("Run FOLA with Csd")
            args2.n_epoch = epoch
            args2.weight = 1
            for csd in csds:
                args2.csd_importance = csd
                fola(args2)
        else:
            print("Run FOLA_prior with Csd")
            args2.n_epoch = epoch
            args2.weight = 0
            for csd in csds:
                args3.csd_importance = csd
                fola_prior(args3)



