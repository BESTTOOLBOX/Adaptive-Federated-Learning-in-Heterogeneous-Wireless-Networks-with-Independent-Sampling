#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy



def Avg(users, w_glob, freq):

    w_avg = copy.deepcopy(w_glob)

    for i, (net_id, u) in enumerate(users.items()):
        w_i = u.model.state_dict()
        if i == 0:
            for key in w_i:
                w_avg[key] = w_i[key] * freq[i]
        else:
            for key in w_i:
                w_avg[key] += w_i[key] * freq[i]
    return w_avg




