#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from data.nlp import mask_tokens
from torch.autograd import Variable


def get_batch(args, source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target



def test_text(net_g, datatest, args):
    net_g.eval()
    data_loader = DataLoader(datatest, batch_size=args.bs)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)

            if idx == 100:
                break

            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        # test_loss /= len(data_loader.dataset)
        # accuracy = 100.00 * float(correct) / len(data_loader.dataset)
        test_loss /= (100 * args.bs)
        accuracy = 100.00 * float(correct) / (100 * args.bs)
        
        return accuracy, test_loss


def test_reddit(net_g, datatest, args, n_tokens):
    net_g.eval()
    total_loss = 0.0
    correct = 0.0

    total_test_words = 0.0
    hidden = net_g.init_hidden(args.bs)
    data_iterator = range(0, datatest.size(0) - 1, args.bptt)
    # data_iterator = random.sample(data_iterator, len(data_iterator) // args.partial_test)
    dataset_size = len(datatest)
    with torch.no_grad():
        for idx, batch in enumerate(data_iterator):
            data, targets = get_batch(args, datatest, batch)
            if args.gpu != -1:
                data, targets = data.to(args.device), targets.to(args.device)


            output, hidden = net_g(data, hidden)
            output_flat = output.view(-1, n_tokens)
            total_loss += len(data) * torch.nn.functional.cross_entropy(output_flat, targets).data
            hidden = tuple([each.data for each in hidden])
            pred = output_flat.data.max(1)[1]
            correct += pred.eq(targets.data).sum().to(dtype=torch.float)
            total_test_words += targets.data.shape[0]

        acc = 100.0 * (correct / total_test_words)
        test_loss = total_loss.item() / (dataset_size - 1)
        acc = acc.item()

        return acc, test_loss