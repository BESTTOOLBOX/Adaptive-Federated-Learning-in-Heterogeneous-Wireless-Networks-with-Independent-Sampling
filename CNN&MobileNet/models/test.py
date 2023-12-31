#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            #data, target = data.cuda(), target.cuda()
            #data, target = data.to(args.device), target.cuda().to(args.device)
            ll = 1
            if ll == 1:
                data, target = data.to(args.device), target.to(args.device)
            else:
                # LSTM
                target = target.to(args.device)
                time_step = 32
                input_size = 32 * 3
                data = data.reshape(-1, time_step, input_size).to(args.device)

        if idx == 20:
            break
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    #test_loss /= len(data_loader.dataset)
    #accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    test_loss /= (20 * args.bs)
    accuracy = 100.00 * float(correct) /  (20 * args.bs)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

import unittest


class TestStringMethods(unittest.TestCase):
    """Test string methods."""

    def test_upper(self) -> None:
        """Testing upper method."""
        self.assertEqual("foo".upper(), "FOO")


if __name__ == "__main__":
    unittest.main()