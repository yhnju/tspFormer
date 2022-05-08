import os
import time
import torch
import datetime
import argparse
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt

from tspformer.transNet import Tspformer
from tspformer.transEncoder import Tspformer_encoder
from tspformer.transDecoder import Tspformer_decoder
from tspformer.sampledAtten import SampledAtten, AttentionLayer

from utils.log import log
from utils.options import get_options
from utils.gpuORcpu import gpuORcpu
from utils.masking import TspMask, SampledMask
from utils.tspLength import compute_tour_length
warnings.filterwarnings("ignore", category=UserWarning)

device = gpuORcpu()
args = get_options()

model_train = Tspformer(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.nb_layers_encoder, args.nb_layers_decoder,
                      args.nb_heads, args.max_len_PE, batchnorm=args.batchnorm)

model_baseline = Tspformer(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.nb_layers_encoder, args.nb_layers_decoder, 
                         args.nb_heads, args.max_len_PE, batchnorm=args.batchnorm)

model_train = model_train.to(device)
model_baseline = model_baseline.to(device)
model_baseline.eval()

optimizer = torch.optim.Adam( model_train.parameters() , lr = args.lr ) 

file, time_stamp = log()
#########################
# Main training loop 
#########################
start_training_time = time.time()
plot_performance_train = []
plot_performance_baseline = []
all_strings = []
epoch_ckpt = 0
tot_time_ckpt = 0

for epoch in range(0,args.nb_epochs):
    epoch += epoch_ckpt
    start = time.time()
    model_train.train() 
    for step in range(1,args.nb_batch_per_epoch+1):    
        # generate a batch of random TSP instances    
        #torch.manual_seed(8989)
        x = torch.rand(args.batch_size, args.nb_nodes, args.dim_input_nodes, device=device) # size(x)=(bsz, nb_nodes, dim_input_nodes) 
      # compute tours for model
        tour_train, sumLogProbOfActions = model_train(x, deterministic=False) 
        
        # size(tour_train)=(bsz, nb_nodes), size(sumLogProbOfActions)=(bsz)
        # compute tours for baseline
        with torch.no_grad():
            tour_baseline, _ = model_baseline(x, deterministic=True)
            #tour_baseline, sumLogProbOfActions = model_baseline(x, deterministic=True)

        # get the lengths of the tours
        L_train = compute_tour_length(x, tour_train) # size(L_train)=(bsz)
        L_baseline = compute_tour_length(x, tour_baseline) # size(L_baseline)=(bsz)
        # backprop
        loss = torch.mean( (L_train - L_baseline)* sumLogProbOfActions )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    time_one_epoch = time.time()-start
    time_tot = time.time()-start_training_time + tot_time_ckpt
    ####################################################
    # Evaluate train  and baseline on random TSP instances
    #####################################################
    model_train.eval()
    mean_tour_length_train = 0
    mean_tour_length_baseline = 0
    for step in range(0,args.nb_batch_eval):
        # generate a batch of random tsp instances   
        x = torch.rand(args.batch_size, args.nb_nodes, args.dim_input_nodes, device=device) 
        # compute tour for model and baseline
        with torch.no_grad():
            tour_train, _ = model_train(x, deterministic=True)
            tour_baseline, _ = model_baseline(x, deterministic=True)
        # get the lengths of the tours
        L_train = compute_tour_length(x, tour_train)
        L_baseline = compute_tour_length(x, tour_baseline)
        # L_tr and L_bl are tensors of shape (bsz,). Compute the mean tour length
        mean_tour_length_train += L_train.mean().item()
        mean_tour_length_baseline += L_baseline.mean().item()
    mean_tour_length_train =  mean_tour_length_train/ args.nb_batch_eval
    mean_tour_length_baseline =  mean_tour_length_baseline/ args.nb_batch_eval
    # evaluate train model and baseline and update if train model is better
    update_baseline = mean_tour_length_train+args.tol < mean_tour_length_baseline
    if update_baseline:
        model_baseline.load_state_dict( model_train.state_dict() )
    #########################################
    # test  baseline model on random TSP instances
    #########################################
    xTest = torch.rand(args.batch_size, args.nb_nodes, args.dim_input_nodes, device=device)
    with torch.no_grad():
        tour_baseline, _ = model_baseline(xTest, deterministic=True)
    mean_tour_length_test = compute_tour_length(xTest, tour_baseline).mean().item()
    #########################################
    # For checkpoint
    #########################################
    plot_performance_train.append([ (epoch+1), mean_tour_length_train])
    plot_performance_baseline.append([ (epoch+1), mean_tour_length_baseline])
    # Compute optimality gap
    if args.nb_nodes==20: gap_train = mean_tour_length_train/3.75- 1.0
    elif args.nb_nodes==50: gap_train = mean_tour_length_train/5.69- 1.0
    elif args.nb_nodes==100: gap_train = mean_tour_length_train/7.76- 1.0
    elif args.nb_nodes==200: gap_train = mean_tour_length_train/10.53- 1.0
    elif args.nb_nodes==500: gap_train = mean_tour_length_train/16.58- 1.0
    elif args.nb_nodes==750: gap_train = mean_tour_length_train/20.14- 1.0
    elif args.nb_nodes==1000: gap_train = mean_tour_length_train/23.12- 1.0
    else: gap_train = -1.0
    # Print and save in txt file
    mystring_min = 'Epoch: {:d}, epoch time: {:.3f}sec, L_train: {:.3f}, L_base: {:.3f}, L_test: {:.3f},\
     gap_train(%): {:.3f}, update: {}'.format(epoch+1, time_one_epoch, mean_tour_length_train,\
      mean_tour_length_baseline, mean_tour_length_test, 100*gap_train, update_baseline) 
    print(mystring_min) # Comment if plot display
    file.write(mystring_min+'\n')
    #########################################
    # Saving checkpoint
    #########################################
    checkpoint_dir = os.path.join("checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save({'epoch': epoch,'time': time_one_epoch,'tot_time': time_tot,'loss': loss.item(),
        'TSP_length': [torch.mean(L_train).item(), torch.mean(L_baseline).item(), mean_tour_length_test],
        'plot_performance_train': plot_performance_train,'plot_performance_baseline': plot_performance_baseline,
        'mean_tour_length_test': mean_tour_length_test,'model_baseline': model_baseline.state_dict(),
        'model_train': model_train.state_dict(),'optimizer': optimizer.state_dict()}, 
        '{}.pkl'.format(checkpoint_dir + "/cpt_" + time_stamp + "-n{}".format(args.nb_nodes)))
