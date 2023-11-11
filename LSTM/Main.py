#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.utils.data.distributed
from utils.options import args_parser
from utils.others import initial_logging, name_save
from models.model_init import init_model
from models.text.RNN import RNNModel
from data.data_init import SSpeare
from models.Fed import Avg
from models.test import test_text, test_reddit
from Client import FL_client_text, FL_client_reddit
from data.Reddit_utils.text_helper import TextHelper
import csv
import copy
import os
from tqdm import tqdm
import threading
import time
import random



def get_user_list(bs):
    device_type=[]
    local_energy_origin=[]
    communication_rate=[]
    communication_time=[]
    communication_energy=[]
    computation_time=[]
    computation_energy=[]
    with open("./device_data/user_list_"+args.model+"_"+args.dataset+".csv","r") as f:
        reader=csv.reader(f)
        header_row=next(reader)
        for row in reader:
            device_type.append(str(row[1]))
            local_energy_origin.append(float(row[2]))
            communication_rate.append(float(row[3]))
            communication_time.append(float(row[4]))
            communication_energy.append(float(0.0))
            computation_time.append(float(row[6]))
            computation_energy.append(float(0.0))
    return device_type, local_energy_origin, communication_rate, communication_time, communication_energy, computation_time, computation_energy


def local_energy_check(idx,local_iter,local_energy_remained,local_energy_origin, energy_threshold,communication_energy,computation_energy):
    if (local_energy_remained[idx]-communication_energy[idx]-computation_energy[idx]*local_iter < local_energy_origin[idx]*energy_threshold):
        return -1
    return 1

def local_energy_update(idx,local_iter,local_energy_remained,communication_energy,computation_energy):
    local_energy_remained[idx]=local_energy_remained[idx]-communication_energy[idx]-computation_energy[idx]*local_iter
    return computation_energy[idx]*local_iter, communication_energy[idx]+computation_energy[idx]*local_iter

def local_train(args,net_glob,idx,local_iter,local_energy_remained,local_energy_origin,energy_threshold,dataset_train,dict_users,communication_energy,computation_energy,g_locals,w_locals,loss_locals,local_loss,len_idxs_users,lr):
    global loss_avg
    global nets_this_round
    global user
    global w_glob
    global ifon_panduan
    global tempTime
    global threadstatecnt
    global totalEnergy
    ifon=local_energy_check(idx,local_iter[idx],local_energy_remained,local_energy_origin, energy_threshold,communication_energy,computation_energy)
    if ifon<0:
        print(str(idx)+" is off")
        local_state[idx]="off"
        threadstatecnt=threadstatecnt+1
        return -1
    if not user:
        if args.dataset == 'shakespeare':
            user = FL_client_text(args=args, dataset=dataset_train, idxs=dict_users[idx])
        else:
            exit("Error: undefined dataset")
    w,loss = user.local_train(net=copy.deepcopy(net_glob).to(args.device),lr=lr, H=local_iter[idx])
    nets_this_round[idx]=w
    g = copy.deepcopy(w)
    for k in g.keys():
        g[k] = w[k] - w_glob[k]
    g_locals.append(copy.deepcopy(g))
    w_locals.append(copy.deepcopy(w))
    loss_locals.append(copy.deepcopy(loss))
    local_loss[idx]=copy.deepcopy(loss)
    local_energy_updated, energycnt=local_energy_update(idx,local_iter[idx],local_energy_remained,communication_energy,computation_energy)
    local_energy_consumed[idx]=copy.deepcopy(local_energy_updated)
    tempTime=max(tempTime,communication_time[idx]*len_idxs_users+computation_time[idx]*local_iter[idx])
    totalEnergy=totalEnergy+energycnt
    loss_avg +=loss
    logger.info(">> user {}: Number of samples {}; Training loss: {:.2f}".format(idx,user.n_sample,loss))
    threadstatecnt=threadstatecnt+1
    return 1

def FedAvg(w_old, w):
    w_avg = copy.deepcopy(w[0])
    w_avg_q = copy.deepcopy(w[0])


    for k in w_avg.keys():
        w_avg[k] = w_avg[k] - w_avg[k]
        w_avg_q[k] = w_avg_q[k] - w_avg_q[k]

        for i in range(0, len(w)):
            w_avg[k] = w[i][k] + w_avg[k]#

        # w_avg[k] = torch.true_divide(w_avg[k], len(w)).cpu()
        w_avg[k] = w_avg[k]/len(w)
        w_avg_q[k] = w_old[k] + w_avg[k]#


    return w_avg_q
    
    
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
    args.lr_opt = 'decay' #'decay, fixed
    args.mloss = 'fix' # 'fix', 'lupa', 'adah'


    logger = initial_logging(args)


    if args.dataset == 'shakespeare':
        # DATA loading & processing
        logger.info(">> Partitioning data")
        dataset_train = SSpeare(train=True)
        dataset_test = SSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        logger.info('>> number of user: ' + str(args.num_users) )
        n_tokens = None

    elif args.dataset == 'reddit':
        # DATA loading & processing
        runner_helper = TextHelper(args=args, current_time=None, params=None, name='text')
        runner_helper.load_data()

        logger.info(">> Partitioning data")
        dataset_train = runner_helper.train_data
        n_tokens = len(runner_helper.corpus.dictionary)
    else:
        exit('Error: Not supported dataset')

    # build model
    logger.info(">> Initialize model")
    net_glob = init_model(args=args, n_tokens=n_tokens)
    w_glob = net_glob.state_dict()

    logger.info(">> Client initialization")
    nets = {net_i: None for net_i in range(args.num_users)}


    INF=2147483647
    # training
    H = args.local_H
    lr = args.lr
    X = np.zeros((args.epochs, 4))
    #record_le = np.zeros((args.epochs - 2, 1))
    PATH = "model.pt"
    loss_train = []
    bs=args.local_bs

    first_round_local_iter=int(args.H0)
    local_age=[ 1 for nn in range(0,args.num_users)]
    local_loss=[ 0 for nn in range(0,args.num_users)]
    local_state=[ "on" for nn in range(0,args.num_users)]
    device_type, local_energy_origin, communication_rate, communication_time, communication_energy, computation_time, computation_energy=get_user_list(bs)
    local_energy_remained=copy.deepcopy(local_energy_origin)############energy_remained
    energy_threshold= args.energy_threshold
    H0=int(args.H0)
    Ha=[[] for nn in range(0,args.num_users)]
    totalEnergy=0
    local_energy_consumed=[0 for nn in range(0,args.num_users)]
    H_old=H0
    totalTime=0
    totaloff=0
    chooseoff=0

    logger.info(args.device)
    logger.info(">> Total communication rounds: " + str(args.epochs))
    s_name, s_acc, = name_save(args)
    logger.info('>>' + s_name)

    os.mkdir("./log/"+args.id)
    
    a_n=[0 for n in range(0,args.num_users)]
    totalWeight=0
    for n in range(0,args.num_users):
        totalWeight=totalWeight+len(dict_users[n])
    for n in range(0,args.num_users):
        a_n[n]=len(dict_users[n])/totalWeight
    q_n=[0 for n in range(0,args.num_users)]
    get_qn(args.sampling,args.num_users,dict_users,args)
    print(q_n)
    
    with open("./log/"+args.id+"/"+args.id+"_qnan.csv","a") as f:
        csvwriter=csv.writer(f)
        csvwriter.writerow(["q_n","a_n"])
        if args.sampling=="full":
            for n in range(0,args.num_users):
                csvwriter.writerow([1,len(dict_users[n])/totalWeight])
        if args.sampling=="uniform":
            for n in range(0,args.num_users):
                csvwriter.writerow([1/args.num_users,len(dict_users[n])/totalWeight])
        if args.sampling=="weight":
            for n in range(0,args.num_users):
                csvwriter.writerow([len(dict_users[n])/totalWeight,len(dict_users[n])/totalWeight])
        if args.sampling=="fix":
            for n in range(0,args.num_users):
                csvwriter.writerow([1/5,len(dict_users[n])/totalWeight])

    for iter in range(args.epochs):
        w_locals, loss_locals, g_locals = [], [] , []
        logger.info(">> " + str(iter))
        offlist=[]
        onlist=[]
        for i in range(0,args.num_users):
            if local_state[i]=="off":
                offlist.append(i)
        onlist=list(set(range(args.num_users))-set(offlist))
        # lr scheduling
        lr = args.lr

        # H policy
        #local_iter = H_policy(args, iter, H, X, record_le)
        # SELECT USER
        #idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        local_iter=[first_round_local_iter for nn in range(0,args.num_users)]
        idxs_users =clientSampling(args.sampling,args.num_users,dict_users,args)
        print(idxs_users)

        with open("./log/"+args.id+"/REWAFL_"+args.id+".txt","a") as f:
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
        with open("./log/"+args.id+"/"+args.id+"_localstate.csv","a") as f:
            csvwriter=csv.writer(f)
            csvwriter.writerow(local_state)
        with open("./log/"+args.id+"/"+args.id+"_localiter.csv","a") as f:
            csvwriter=csv.writer(f)
            csvwriter.writerow(local_iter)
        

        nets_this_round = {k: None for k in idxs_users}

        loss_avg = 0
        # local training
        ifon_panduan=0
        tempTime=0
        threadtotal=[]
        threadstatecnt=0
        parallelnum=20
        periodcnt=0
        restparallel=len(idxs_users)
        with tqdm(total=len(idxs_users)) as bar:
            for idx, user in nets_this_round.items():
                if restparallel<parallelnum:
                    parallelnum=restparallel
                if args.grad_qn==1:
                    newlr=lr/(q_n[idx]/a_n[idx])
                else:
                    newlr=lr
                local_thread=threading.Thread(target=local_train,args=(args,net_glob,idx,local_iter,local_energy_remained,local_energy_origin,energy_threshold,dataset_train,dict_users,communication_energy,computation_energy,g_locals,w_locals,loss_locals,local_loss,len(idxs_users),newlr))
                local_thread.start()
                threadtotal.append(local_thread)
                periodcnt=periodcnt+1
                if periodcnt>=parallelnum:
                    while (threadstatecnt<parallelnum):
                        time.sleep(1)
                    threadstatecnt=0
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
        chooseoff=0
        totaloff=totaloff/args.num_users
        old_model = copy.deepcopy(w_glob)
        if (len(g_locals)>0):
            w_glob = FedAvg(old_model, g_locals)
            net_glob.load_state_dict(w_glob)
            w_glob = net_glob.state_dict()
            w_glob=w_glob
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_train.append(loss_avg)
        else:
            if len(loss_train)==0:
                loss_train.append(0)
            loss_train.append(loss_train[len(loss_train)-1])

        H_old = local_iter


        if iter % 1 == 0:
            if args.dataset == 'shakespeare':
                acc_test, loss_test_1 = test_text(net_glob, dataset_test, args)
            elif args.dataset == 'reddit':
                acc_test, loss_test_1 = test_reddit(net_glob, runner_helper.test_data, args, n_tokens)
            else:
                exit("Error: undefined dataset")
            print(iter)
            print("Testing accuracy: {:.6f}".format(acc_test))
            print("Testing Loss: {:.6f}".format(loss_test_1))
            logger.info("Testing accuracy: {:.6f}".format(acc_test))
            logger.info("Testing Loss: {:.6f}".format(loss_test_1))

            X[iter // 1, 0] = acc_test
            X[iter // 1, 1] = loss_test_1
            X[iter // 1, 2] = loss_avg
            # acc_train, loss_train_1 = test_text(net_glob, dataset_train, args)
            # print("Training accuracy: {:.2f}".format(acc_train))
            # print("Training Loss: {:.2f}".format(loss_train_1))
            # X[iter // 1, 2] = acc_train
            # X[iter // 1, 3] = loss_train_1

        with open("./log/"+args.id+"/"+args.id+"_accloss.csv","a") as f:
            csvwriter=csv.writer(f)
            csvwriter.writerow([acc_test,loss_test_1,totalEnergy,totalTime,chooseoff,totaloff])

#    torch.save({'epoch': args.epochs,
#                'model_state_dict': net_glob.state_dict(),
#                # 'optimizer_state_dict': optimizer.state_dict(),
#                'loss': loss_test_1,
#                }, PATH)

    pd_data1 = pd.DataFrame(X)
    pd_data1.to_csv(s_acc)


    loss = X[:, 2]
    plt.figure()
    plt.plot(range(len(X)), loss)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}.png'.format(args.dataset, args.model, args.epochs))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_text(net_glob, dataset_train, args)
    acc_test, loss_test = test_text(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Training Loss: {:.5f}".format(loss_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    print("Testing Loss: {:.5f}".format(loss_test))


