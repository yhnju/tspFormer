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

from concorde.tsp import TSPSolver # !pip install -e pyconcorde
from torch.distributions.categorical import Categorical

from utils.tspLength import compute_tour_length
from utils.options import get_options
from utils.gpuORcpu import gpuORcpu
from utils.log import log
from utils.plotTSP import plot_tsp
from utils.positionEncode import generate_positional_encoding

from tspformer.transNet import Tspformer
from tspformer.transEncoder import Tspformer_encoder
from tspformer.transDecoder import Tspformer_decoder
from tspformer.sampledAtten import SampledAtten, AttentionLayer

from search.greedy import greedySearch
from search.beam import beam0, beam1, beamGreaterEq2
warnings.filterwarnings("ignore", category=UserWarning)

device = gpuORcpu()
args = get_options()


class TspformerSearch(nn.Module): 
    def __init__(self, dim_input_nodes, dim_emb, dim_ff, nb_layers_encoder, nb_layers_decoder, nb_heads, max_len_PE,
                 batchnorm=True):
        super(TspformerSearch, self).__init__()
        self.dim_emb = dim_emb
        # input embedding layer
        self.input_emb = nn.Linear(dim_input_nodes, dim_emb)
        # encoder layer
        self.encoder = Tspformer_encoder(nb_layers_encoder, dim_emb, nb_heads, dim_ff, batchnorm)
        # vector to start decoding 
        self.start_placeholder = nn.Parameter(torch.randn(dim_emb))
        # decoder layer
        self.decoder = Tspformer_decoder(dim_emb, nb_heads, nb_layers_decoder)
        self.WK_att_decoder = nn.Linear(dim_emb, nb_layers_decoder* dim_emb) 
        self.WV_att_decoder = nn.Linear(dim_emb, nb_layers_decoder* dim_emb) 
        self.PE = generate_positional_encoding(dim_emb, max_len_PE)        
        
    def forward(self, x, B, greedy, beamsearch):
        # some parameters
        bsz = x.shape[0]
        nb_nodes = x.shape[1]
        zero_to_bsz = torch.arange(bsz, device=x.device) # [0,1,...,bsz-1]
        # For beam search
        zero_to_B = torch.arange(B, device=x.device) # [0,1,...,B-1]
        # input embedding layer
        h = self.input_emb(x) # size(h)=(bsz, nb_nodes, dim_emb)
        # concat the nodes and the input placeholder that starts the decoding
        h = torch.cat([h, self.start_placeholder.repeat(bsz, 1, 1)], dim=1) # size(start_placeholder)=(bsz, nb_nodes+1, dim_emb)
        # encoder layer
        h_encoder = self.encoder(h) # size(h)=(bsz, nb_nodes+1, dim_emb)

        # key and value for decoder    
        K_att_decoder = self.WK_att_decoder(h_encoder) # size(K_att)=(bsz, nb_nodes+1, dim_emb*nb_layers_decoder)
        V_att_decoder = self.WV_att_decoder(h_encoder) # size(V_att)=(bsz, nb_nodes+1, dim_emb*nb_layers_decoder)
        #print('K_att_decoder.shape=torch.Size([4, 7, 16])',K_att_decoder.shape,'\nV_att_decoder.shape=torch.Size([4, 7, 16])',V_att_decoder.shape)
        # starting node in tour
        self.PE = self.PE.to(x.device)
        # For beam search
        tours_greedy = torch.zeros(2, nb_nodes, device=x.device)
        tours_beamsearch = torch.zeros(2, nb_nodes, device=x.device)
        scores_greedy = torch.zeros(2, device=x.device)
        scores_beamsearch = torch.zeros(2, device=x.device)
        # Greedy search
        if greedy:
            #print('Greedy decoding')
            tours_greedy, scores_greedy = greedySearch(h_encoder,self.decoder,x,B)
        # Beamsearch
        if beamsearch:
            #print('Beam search decoding')
            # clear key and val stored in the decoder
            self.decoder.reset_selfatt_keys_values() 
            K_att_decoder_tmp = K_att_decoder # size(K_att_decoder_tmp)=(bsz, nb_nodes+1, dim_emb*nb_layers_decoder)
            V_att_decoder_tmp = V_att_decoder # size(V_att_decoder_tmp)=(bsz, nb_nodes+1, dim_emb*nb_layers_decoder)
            for t in range(nb_nodes):
                #print('t: {}, GPU reserved mem: {:.2f}, GPU allocated mem: {:.2f}'.format(t,torch.cuda.memory_reserved(0)/1e9,
                #    torch.cuda.memory_allocated(0)/1e9))
                if t==0: # at t=0, there are at most B_{t=0}=nb_nodes beams
                    K_att_decoder, V_att_decoder,h_t, mask_visited_nodes, sum_scores ,tours = \
                    beam0(h_encoder,self.decoder,x,B,t,K_att_decoder,V_att_decoder, K_att_decoder_tmp,V_att_decoder_tmp)
                elif t==1: # at t=1, there are at most B_{t=1}=nb_nodes^2 beams
                    # compute probability over the next node in the tour
                    K_att_decoder, V_att_decoder,h_t, mask_visited_nodes, sum_scores, tours = \
                    beam1(h_encoder,self.decoder,x,B,t,h_t,mask_visited_nodes,K_att_decoder,
                     V_att_decoder,sum_scores,tours,K_att_decoder_tmp,V_att_decoder_tmp)
                else: # at t>=2, we arbitrary decide to have at most B_{t>=2}=nb_nodes^2 beams
                    # compute probability over the next node in the tour
                    tours_beamsearch, scores_beamsearch = beamGreaterEq2(h_encoder,self.decoder,x,B,t,h_t,
                        mask_visited_nodes, K_att_decoder, V_att_decoder, sum_scores, tours)
        return tours_greedy, tours_beamsearch, scores_greedy, scores_beamsearch
###########################################################
# Instantiate a training network and a baseline network
###########################################################
model_baseline = TspformerSearch(args.dim_input_nodes, args.dim_emb, args.dim_ff, args.nb_layers_encoder, args.nb_layers_decoder,
                         args.nb_heads, args.max_len_PE, batchnorm=args.batchnorm)
model_baseline = model_baseline.to(device)
model_baseline.eval()
###################
# Load checkpoint
###################
checkpoint_file = "checkpoint/cpt_220112130550-n20.pkl" 
checkpoint = torch.load(checkpoint_file, map_location=device)
epoch_ckpt = checkpoint['epoch'] + 1
tot_time_ckpt = checkpoint['tot_time']
plot_performance_train = checkpoint['plot_performance_train']
plot_performance_baseline = checkpoint['plot_performance_baseline']
model_baseline.load_state_dict(checkpoint['model_baseline'])
print('Load checkpoint file={:s}\n  Checkpoint at epoch= {:d} and time={:.3f}min\n'.format(checkpoint_file,epoch_ckpt-1,tot_time_ckpt/60))
del checkpoint
mystring_min = 'Epoch: {:d}, L_train: {:.3f}, L_base: {:.3f}\n'.format(epoch_ckpt,
                plot_performance_train[-1][1], plot_performance_baseline[-1][1]) 
print(mystring_min) 
bsz = 32; nb_nodes = 20; B = 10; greedy = True; beamsearch = True 
# nb_nodes = 100; B = 100; nb_nodes = 100; B = 1000; nb_nodes = 100; B = 3000
# nb_nodes = 200; B = 100; nb_nodes = 200; B = 1000
seedN = torch.manual_seed(12)
print('seed=',seedN)
x = torch.rand(bsz, nb_nodes, 2, device = device)
#print('x=',x)
with torch.no_grad():
    tours_greedy, tours_beamsearch, scores_greedy, scores_beamsearch = model_baseline(x, B, True, True)
    # greedy
    L_greedy = compute_tour_length(x, tours_greedy)
    mean_tour_length_greedy = L_greedy.mean().item()  
    mean_scores_greedy = scores_greedy.mean().item()    
    # beamsearch
    #print('\ntours_beamsearch=',tours_beamsearch)
    tours_beamsearch = tours_beamsearch.view(bsz*B, nb_nodes)
    #print('\ntours_beamsearch2=',tours_beamsearch)
    x_beamsearch = x.repeat_interleave(B,dim=0)
    #print('\nx_beamsearch=',x_beamsearch)
    L_beamsearch = compute_tour_length(x_beamsearch, tours_beamsearch)
    #print('\nL_beamsearch=',L_beamsearch)
    L_beamsearch = L_beamsearch.view(bsz, B)
    #print('\nL_beamsearch2=',L_beamsearch)
    L_beamsearch, idx_min = L_beamsearch.min(dim=1)
    #print('\nL_beamsearch3=',L_beamsearch,'\nidx_min=',idx_min)
    tours_beamsearch = tours_beamsearch.view(bsz, B, nb_nodes)
    torch.cuda.empty_cache() # free GPU reserved memory 
    #print('GPU reserved mem: {:.2f}, GPU allocated mem: {:.2f}'.format(torch.cuda.memory_reserved(0)/1e9,torch.cuda.memory_allocated(0)/1e9))
print('\nL_greedy=',L_greedy)
print('\nL_beamsearch=',L_beamsearch)
tours = []
for b in range(bsz):
    tours.append(tours_beamsearch[b,idx_min[b],:])
print('\nbeam search tours=',tours)
tours_beamsearch = torch.stack(tours, dim=0)
print('\ntours_beamsearch=',tours_beamsearch)

plot_tsp(x, tours_greedy, plot_concorde=True)
plot_tsp(x, tours_beamsearch, plot_concorde=True)
