#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, batch_size = None, option = True):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss().cuda()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=option)
        #self.ldr_train = torch.utils.data.DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=option, num_workers = 4)

    def train(self, net, lr, ep, local_iter):
        net.train()
        # train and update
        #self.args.lr *= 0.999
        #if self.args.lr < 0.000001:
        #    self.args.lr = 0.000001
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay= 1e-4)
        #scaler=torch.cuda.amp.GradScaler()
        epoch_loss = []
        for iter in range(ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                if batch_idx == local_iter:
                    break
                images, labels = images.to(self.args.device), labels.to(self.args.device)


                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                #with torch.cuda.amp.autocast():
                #    log_probs=net(images)
                #    loss=self.loss_func(log_probs,labels)
                    
                #scaler.scale(loss).backward()
                #scaler.unscale_(optimizer)
                #scaler.step(optimizer)
                #scaler.update()
                net.zero_grad()
                loss.backward()
                optimizer.step()


                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        net=net.cpu()
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

