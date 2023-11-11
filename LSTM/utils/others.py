import os
import logging
import datetime

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def nu_classes(args):

    if args.dataset in {'mnist', 'cifar', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 26
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset == 'xray':
        n_classes = 2
    elif args.dataset == 'wiki_text' or args.dataset == 'shakespeare':
        n_classes = 80
    else:
        exit('Error: unrecognized dataset')

    return n_classes



def name_save(args):
    s_name = '{}_{}_{}_B{}_E{}_N{}_LR{}'.format(args.mloss, args.model, args.dataset, args.local_bs, args.local_H,
                                                     args.num_users, args.lr)

    s_acc =  s_name + '.csv'
    return s_name, s_acc



def initial_logging(args):
    # logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    mkdirs(args.logdir)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
    return logger

