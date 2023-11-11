#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models.text.lstm import ModelLSTMShakespeare


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class FL_client(object):
    def __init__(self, args):
        self.args = args
        if self.args.dataset == 'shakespeare':
            self.model = ModelLSTMShakespeare(args=args).to(args.device)
        elif self.args.dataset == 'reddit':
            self.model = None
        self.ep = 1
        self.loss_func = nn.CrossEntropyLoss().to(self.args.device)

    def assign_model(self, weights):
        self.model.load_state_dict(copy.deepcopy(weights))



class FL_client_text(FL_client):
    def __init__(self, args, dataset=None, idxs=None):
        super(FL_client_text, self).__init__(args)
        self.ldr_train = DataLoader(DatasetSplit(dataset, list(idxs)),
                                    batch_size=self.args.local_bs,
                                    shuffle=True,
                                    )
        self.n_sample = len(self.ldr_train)

    def local_train(self,net, lr, H):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        epoch_loss = []
        total_gradient_calculate=0
        for iter in range(H):
            batch_loss = []
            for batch_idx, (data, labels) in enumerate(self.ldr_train):
                total_gradient_calculate=total_gradient_calculate+1
                data, labels = data.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                log_probs = net(data)

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                if total_gradient_calculate>=H:
                    break
            

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if total_gradient_calculate>=H:
                break
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)



class FL_client_reddit(FL_client):
    def __init__(self, args, dataset=None, ntokens=None):
        super(FL_client_reddit, self).__init__(args)
        # we choose the first 90% of each participant's local data as their local training set
        trunk = len(dataset) // 100 * (100 - self.args.local_test_perc)
        self.ldr_train = dataset[:trunk]
        self.ntokens = ntokens
        self.n_sample = len(self.ldr_train)

    def get_batch(self, source, i):
        seq_len = min(self.args.bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)
        return data, target

    def local_train(self, lr, H):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        hidden = self.model.init_hidden(self.args.local_bs)
        epoch_loss = 0.
        data_iterator = range(0, self.ldr_train.size(0) - 1, self.args.bptt)
        for iter in range(H):
            total_loss = 0.
            batch_num = 0
            for batch_id, batch in enumerate(data_iterator):
                batch_num += 1
                optimizer.zero_grad()
                data, targets = self.get_batch(self.ldr_train, batch)
                hidden = tuple([each.data for each in hidden])
                output, hidden = self.model(data, hidden)
                loss = self.loss_func(output.view(-1, self.ntokens), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                optimizer.step()
                total_loss += loss.item()
            if batch_num > 0:
                epoch_loss += (total_loss/ batch_num)
        return epoch_loss / H