#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=40, help="rounds of training")#1172
    parser.add_argument('--num_users', type=int, default=128, help="number of users: K")
    parser.add_argument('--local_H', type=int, default=3, help="the number of local iteration: H")
    parser.add_argument('--local_bs', type=int, default=20, help="local batch size: B")

    parser.add_argument('--bs', type=int, default=10, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.8, help="learning rate") # 0.8 for shakespeare 40 for reddit
    parser.add_argument('--lr_opt', type=str, default='fixed', help='scheduler')
    parser.add_argument('--mloss', type=str, default='adah',
                        choices=['fix', 'adah', 'lupa'], help='H policy')

    # model arguments
    parser.add_argument('--model', type=str, default='lstm',
                        choices=['rnn', 'lstm'], help='model name')

    # rnn
    parser.add_argument('--emsize', type=int, default=200, help='embedding dimension')
    parser.add_argument("--nhid", type=int, default=200, help="RNN hidden unit dimensionality")
    parser.add_argument("--nlayers", type=int, default=2, help="Number of RNN layers")
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--tied', type=bool, default=True, help="Use tied input/output embedding weights: 1 for true")

    # dataset arguments
    parser.add_argument('--dataset', type=str, default='shakespeare',
                        choices=['shakespeare', 'reddit'], help="name of dataset") #cifar tiny_image
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of images")

    #reddit
    parser.add_argument('--local_test_perc', type=int, default=10, help='percentage of local test data')
    parser.add_argument("--partial_test", type=int, default=5, help="test subset")
    parser.add_argument("--bptt", type=int, default=64)
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')

    # other arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    
    parser.add_argument('--id',type=str)
    parser.add_argument('--energy_threshold',default=0.0,type=float)
    parser.add_argument('--H0',type=float)
    parser.add_argument('--sampling',type=str)
    parser.add_argument('--grad_qn',type=int)

    args = parser.parse_args()
    return args
