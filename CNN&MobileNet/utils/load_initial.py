from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from models.Nets import MLP, CNNMnist, CNNCifar, HARmodel
#from models.MobileNetV2Real import MobileNetV2
from models.MobileNetV2 import MobileNet as MobileNetV2
from models.vgg19 import vgg19
from models.lstm import lstm
from utils.options import args_parser
import dill

class HARDataset(Dataset):
    def __init__(self, data_path, train=True, transform=None):
        if train==True:
            with open(data_path+"/train_x.pkl","rb") as f:
                self.data=dill.load(f)
            with open(data_path+"/train_y.pkl","rb") as f:
                self.targets=dill.load(f)
        else:
            with open(data_path+"/test_x.pkl","rb") as f:
                self.data=dill.load(f)
            with open(data_path+"/test_y.pkl","rb") as f:
                self.targets=dill.load(f)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform is not None:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

def load_data(args):

    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            #dict_users = mnist_iid(dataset_train, args.num_users)
            with open("./data/data_saved/mnist_iid_dataset.pkl","rb") as f:
                dict_users=dill.load(f)
        else:
            #dict_users = mnist_noniid(dataset_train, args.num_users)
            with open("./data/data_saved/mnist_noniid_dataset.pkl","rb") as f:
                dict_users=dill.load(f)

    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                      std=[0.229, 0.224, 0.225])])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        print(dataset_test.data.shape)
        if args.iid:
            #dict_users = cifar_iid(dataset_train, args.num_users)
            with open("./data/data_saved/cifar_iid_dataset.pkl","rb") as f:
                dict_users=dill.load(f)
        else:
            #dict_users = cifar_noniid(dataset_train, args.num_users)
            with open("./data/data_saved/cifar_noniid_dataset.pkl","rb") as f:
                dict_users=dill.load(f)
    elif args.dataset == 'har':
        dataset_train=HARDataset('./data/har',train=True,transform=None)
        dataset_test=HARDataset('./data/har',train=False,transform=None)
        with open("./data/data_saved/har_dataset.pkl","rb") as f:
            dict_users=dill.load(f)
    else:
        exit('Error: unrecognized dataset')
    print(dataset_train)
    print(dataset_test)
    return dict_users, dataset_train, dataset_test


def load_model(args, img_size):
    # build model
    # CNNCifar = CNNCifar(args=args).cuda()
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
        #net_glob = CNNCifar(args=args)

    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
       # net_glob = CNNMnist(args=args)
    
    elif args.model == 'cnn' and args.dataset == 'har':
        net_glob = HARmodel(input_channel=1, num_classes=6).to(args.device)

    elif args.model == 'mobile' and args.dataset == 'cifar':
        net_glob = MobileNetV2(args=args).to(args.device)
        #net_glob = MobileNetV2(args=args)

    elif args.model == 'vgg' and args.dataset == 'cifar':
        net_glob = vgg19(args=args).to(args.device)
        #net_glob = vgg19(args=args)

    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        #net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes)

    else:
        exit('Error: unrecognized model')

    return net_glob

if __name__ == '__main__':
    args = args_parser()
    load_data(args)
