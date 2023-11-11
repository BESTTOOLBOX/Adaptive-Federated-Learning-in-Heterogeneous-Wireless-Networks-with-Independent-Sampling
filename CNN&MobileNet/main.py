#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
import time

from utils.load_initial import load_data, load_model
from utils.options import args_parser
from models.Update import LocalUpdate

from models.Fed import FedAvg
from models.test import test_img
import torch.utils.data.distributed
import os
import csv
from tqdm import tqdm
import threading
import dill
import random



def get_user_list(bs):
    device_type=[]
    local_energy_origin=[]
    communication_rate=[]
    communication_time=[]
    communication_energy=[]
    computation_time=[]
    computation_energy=[]
    with open("./device_data/user_list_"+args.model+"_"+args.dataset+args.datasetnum+".csv","r") as f:
        reader=csv.reader(f)
        header_row=next(reader)
        for row in reader:
            device_type.append(str(row[1]))
            local_energy_origin.append(float(row[2]))
            communication_rate.append(float(row[3]))
            communication_time.append(float(row[4])/4)
            communication_energy.append(float(0.0))
            computation_time.append(float(row[6])*bs)
            computation_energy.append(float(0.0))
    return device_type, local_energy_origin, communication_rate, communication_time, communication_energy, computation_time, computation_energy


def local_energy_check(idx,local_iter,local_energy_remained,local_energy_origin, energy_threshold):
    if (local_energy_remained[idx]-communication_energy[idx]-computation_energy[idx]*local_iter < local_energy_origin[idx]*energy_threshold):
        return -1
    return 1

def local_energy_update(idx,local_iter,local_energy_remained,communication_energy,computation_energy):
    local_energy_remained[idx]=local_energy_remained[idx]-communication_energy[idx]-computation_energy[idx]*local_iter
    return local_energy_remained[idx], communication_energy[idx]+computation_energy[idx]*local_iter

def local_train(args,dataset_train,dict_users,idx,bs,option,net_glob,lr,local_iter,local_energy_remained,local_energy_origin,energy_threshold,communication_energy,computation_energy,g_locals,w_locals,loss_locals,local_loss,len_idxs_users):
    global ifon_panduan
    global tempTime
    global threadstatecnt
    global totalEnergy
    global w_glob
    global paralleloff
    ifon=local_energy_check(idx,local_iter,local_energy_remained,local_energy_origin, energy_threshold)
    if ifon<0:
        print(str(idx)+" is off")
        local_state[idx]="off"
        paralleloff=paralleloff+1
        return -1
    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], batch_size = bs, option= option)
    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), lr=lr, ep=1, local_iter=local_iter)
    g = copy.deepcopy(w)
    for k in g.keys():
        g[k] = w[k] - w_glob[k]


    local_energy_updated, energycnt=local_energy_update(idx,local_iter,local_energy_remained,communication_energy,computation_energy)
    tempTime=max(tempTime,communication_time[idx]*len_idxs_users+computation_time[idx]*local_iter)
    totalEnergy=totalEnergy+energycnt
    g_locals.append(copy.deepcopy(g))
    w_locals.append(copy.deepcopy(w))
    loss_locals.append(copy.deepcopy(loss))
    local_loss[idx]=copy.deepcopy(loss)
    threadstatecnt=threadstatecnt+1
    return 1
    

def generate_binary_distribution(p):
    if random.random() < p:
        return 1
    else:
        return 0

def clientSampling(sampling, N, dict_users,args):
    global q_n
    ans=[]
    for n in range(0,N):
        if generate_binary_distribution(q_n[n])==1:
            ans.append(n)
    return ans

def get_qn(sampling, N, dict_users,args):
    global q_n
    if sampling=="full":
        for n in range(0,N):
            q_n[n]=1
        return 0
    if sampling=="uniform":
        for n in range(0,N):
            q_n[n]=1/N
        return 0
    if sampling=="weight":
        totalWeight=0
        for n in range(0,N):
            totalWeight=totalWeight+len(dict_users[n])
        for n in range(0,N):
            q_n[n]=len(dict_users[n])/totalWeight
        return 0
    if sampling=="fix":
        for n in range(0,N):
            q_n[n]=1/5
        return 0
    if sampling=="proposed":
        cnt=0
        with open("./"+str(args.model)+"_output.csv","r") as f:
            reader=csv.reader(f)
            for row in reader:
                q_n[cnt]=float(row[0])
                cnt=cnt+1
        return 0
    return 1

    

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.epochs)
    print(args.iid)
    dict_users, dataset_train, dataset_test = load_data(args)
    img_size = dataset_train[0][0].shape

    net_glob = load_model(args, img_size)
    net_glob.train()
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    lr = args.lr
    batch_num = 0
    bs = args.local_bs
    option = True
    X = np.zeros((args.epochs, 6))
    first_round_local_iter=args.H0
    totalTime=0

    INF=2147483647
    record_le = []
    PATH = "model.pt"
    local_loss=[ 0 for nn in range(0,args.num_users)]
    local_state=[ "on" for nn in range(0,args.num_users)]
    device_type, local_energy_origin, communication_rate, communication_time, communication_energy, computation_time, computation_energy=get_user_list(bs)
    local_energy_remained=copy.deepcopy(local_energy_origin)############energy_remained
    energy_threshold= args.energy_threshold
    m = max(int(args.frac * args.num_users), 1)
    os.mkdir("./log/"+args.id)
    totalEnergy=0
    totaloff=0
    chooseoff=0
    offlist=[]
    onlist=[]
    a_n=[0 for n in range(0,args.num_users)]
    totalWeight=0
    for n in range(0,args.num_users):
        totalWeight=totalWeight+len(dict_users[n])
    for n in range(0,args.num_users):
        a_n[n]=len(dict_users[n])/totalWeight
    q_n=[0 for n in range(0,args.num_users)]
    get_qn(args.sampling,args.num_users,dict_users,args)
    print(q_n)
    
    for iter in range(args.epochs):
        offlist=[]
        onlist=[]
        for i in range(0,args.num_users):
            if local_state[i]=="off":
                offlist.append(i)
        onlist=list(set(range(args.num_users))-set(offlist))
        #onlist=range(0,args.num_users)

        lr = args.lr
            
        w_locals, loss_locals, g_locals = [], [] , []
        idxs_users = clientSampling(args.sampling,args.num_users,dict_users,args)
        print(idxs_users)
        local_iter=first_round_local_iter


        with open("./log/"+args.id+"/"+args.id+".txt","a") as f:
            f.write("iter=")
            f.write(str(iter))
            f.write("\nidxs_users\n")
            f.write(str(idxs_users))
            f.write("\nlocal_energy_remained\n")
            f.write(str(local_energy_remained))
            f.write("\nlocal_state\n")
            f.write(str(local_state))
            
        with open("./log/"+args.id+"/"+args.id+"_energyRemained.csv","a") as f:
            csvwriter=csv.writer(f)
            csvwriter.writerow(local_energy_remained)
        with open("./log/"+args.id+"/"+args.id+"_idxsusers.csv","a") as f:
            csvwriter=csv.writer(f)
            csvwriter.writerow(idxs_users)

        #######################################################################
        time_star = time.time()
        ifon_panduan=0
        tempTime=0
        threadtotal=[]
        threadstatecnt=0
        parallelnum=len(idxs_users)
        periodcnt=0
        paralleloff=0
        chooseoff=0
        restparallel=len(idxs_users)
        with tqdm(total=len(idxs_users)) as bar:
            for idx in idxs_users:
                if restparallel<parallelnum:
                    parallelnum=restparallel
                if args.grad_qn==1:
                    newlr=lr/(q_n[idx]/a_n[idx])
                else:
                    newlr=lr
                local_thread=threading.Thread(target=local_train,args=(args,dataset_train,dict_users,idx,bs,option,net_glob,newlr,local_iter,local_energy_remained,local_energy_origin,energy_threshold,communication_energy,computation_energy,g_locals,w_locals,loss_locals,local_loss,len(idxs_users)))
                local_thread.start()
                threadtotal.append(local_thread)
                periodcnt=periodcnt+1
                if periodcnt>=parallelnum:
                    while (threadstatecnt<parallelnum-paralleloff):
                        time.sleep(1)
                    threadstatecnt=0
                    chooseoff=chooseoff+paralleloff
                    paralleloff=0
                    for i in threadtotal:
                        i.join()
                    threadtotal=[]
                    restparallel=restparallel-periodcnt
                    periodcnt=0
                bar.update(1)
        totalTime=totalTime+tempTime
        with open("./log/"+args.id+"/"+args.id+"_localstate.csv","a") as f:
            csvwriter=csv.writer(f)
            csvwriter.writerow(local_state)
        totaloff=0
        for i in local_state:
            if i=="off":
                totaloff=totaloff+1
        totaloff=totaloff/args.num_users
        if len(g_locals)>0:
            # update global weights
            old_model = copy.deepcopy(w_glob)
            w_glob = FedAvg(old_model, g_locals)

            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)
            w_glob = net_glob.state_dict()

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_train.append(loss_avg)
        else:
            if len(loss_train)==0:
                loss_train.append(0)
            loss_train.append(loss_train[len(loss_train)-1])

        H_old = local_iter

        if iter % 1 == 0:
            acc_test, loss_test_1 = test_img(net_glob, dataset_test, args)
            print(iter)
            print("Testing accuracy: {:.2f}".format(acc_test))
            print("Testing Loss: {:.5f}".format(loss_test_1))
            X[iter // 1, 0] = acc_test
            X[iter // 1, 1] = loss_test_1


            acc_train, loss_train_1 = test_img(net_glob, dataset_train, args)
            print("Training accuracy: {:.2f}".format(acc_train))
            print("Training Loss: {:.5f}".format(loss_train_1))
            X[iter // 1, 2] = acc_train
            X[iter // 1, 3] = loss_train_1
            X[iter // 1, 4] = totalEnergy
            X[iter // 1, 5] = totalTime
        with open("./log/"+args.id+"/"+args.id+"_accloss.csv","a") as f:
            csvwriter=csv.writer(f)
            csvwriter.writerow([acc_test,loss_test_1,acc_train,loss_train_1,totalEnergy,totalTime,chooseoff,totaloff])

    pd_data1 = pd.DataFrame(X)
    s1 = './log/'+args.id+'/'+args.id+'_result.csv'
    pd_data1.to_csv(s1)

    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./log/'+args.id+'/'+args.id+'_trainloss.png')

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Training Loss: {:.5f}".format(loss_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    print("Testing Loss: {:.5f}".format(loss_test))

