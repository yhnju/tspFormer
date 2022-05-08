import os
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from tspformer.transEncoder import Tspformer_encoder
from tspformer.transDecoder import Tspformer_decoder, generate_positional_encoding


class Tspformer(nn.Module): 
    def __init__(self, dim_input_nodes, dim_emb, dim_ff, nb_layers_encoder, nb_layers_decoder, nb_heads, max_len_PE,
                batchnorm=True):
        super(Tspformer, self).__init__()
        self.dim_emb = dim_emb
        # input embedding layer
        self.input_emb = nn.Linear(dim_input_nodes, dim_emb)
        #self.input_emb = nn.Linear(dim_input_nodes, dim_emb)
        # encoder layer
        self.encoder = Tspformer_encoder(nb_layers_encoder, dim_emb, nb_heads, dim_ff, batchnorm)
        #self.encoder = Encoder(128, 8, 8, 8, attention_size = None, dropout = 0.3, chunk_mode = None)
 
        # vector to start decoding 
        self.start_placeholder = nn.Parameter(torch.randn(dim_emb))
        # decoder layer
        self.decoder = Tspformer_decoder(dim_emb, nb_heads, nb_layers_decoder)
        #self.decoder = Decoder(128, 8, 8, 8, attention_size = None, dropout = 0.3, chunk_mode = None)

        self.WK_att_decoder = nn.Linear(dim_emb, nb_layers_decoder * dim_emb) 
        self.WV_att_decoder = nn.Linear(dim_emb, nb_layers_decoder * dim_emb) 
        self.PE = generate_positional_encoding(dim_emb, max_len_PE)        

    def forward(self, x, deterministic=False):
        # some parameters
        bsz = x.shape[0]
        nb_nodes = x.shape[1]
        zero_to_bsz = torch.arange(bsz, device=x.device) # [0,1,...,bsz-1]

        #x = batch_knn_distance(bsz, nb_nodes, x.cpu().numpy(), self.knn).to(x.device)
        # input embedding layer
        h = self.input_emb(x) # size(h)=(bsz, nb_nodes, dim_emb)
        mask_visited_nodes = torch.zeros(bsz, nb_nodes+1, device=x.device).bool() # False
        # concat the nodes and the input placeholder that starts the decoding
        h = torch.cat([h, self.start_placeholder.repeat(bsz, 1, 1)], dim=1) # size(start_placeholder)=(bsz, 1, dim_emb)
        # encoder layer
        #h_encoder = self.encoder(h,mask_visited_nodes) # size(h)=(bsz, nb_nodes+1, dim_emb)
        # encoder layer
        h_encoder = self.encoder(h) # size(h)=(bsz, nb_nodes+1, dim_emb),Transformer_encoder_net
        # list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
        tours = []
        # list that will contain Float tensors of shape (bsz,) that gives the neg log probs of the choices made at time t
        sumLogProbOfActions = []
        # key and value for decoder    
        K_att_decoder = self.WK_att_decoder(h_encoder) # size(K_att)=(bsz, nb_nodes+1, dim_emb*nb_layers_decoder)
        V_att_decoder = self.WV_att_decoder(h_encoder) # size(V_att)=(bsz, nb_nodes+1, dim_emb*nb_layers_decoder)

        # input placeholder that starts the decoding
        self.PE = self.PE.to(x.device)
        idx_start_placeholder = torch.Tensor([nb_nodes]).long().repeat(bsz).to(x.device)
        h_start = h_encoder[zero_to_bsz, idx_start_placeholder, :] + self.PE[0].repeat(bsz,1) # size(h_start)=(bsz, dim_emb)
        # initialize mask of visited cities
        #mask_visited_nodes = torch.zeros(bsz, nb_nodes+1, device=x.device).bool() # False
        mask_visited_nodes[zero_to_bsz, idx_start_placeholder] = True
        # clear key and val stored in the decoder
        self.decoder.reset_selfatt_keys_values()

        # construct tour recursively
        h_t = h_start
        for t in range(nb_nodes):
            # compute probability over the next node in the tour
            prob_next_node = self.decoder(h_t, K_att_decoder, V_att_decoder, mask_visited_nodes) 

            # choose node with highest probability or sample with Bernouilli 
            if deterministic:
                idx = torch.argmax(prob_next_node, dim=1) # size(query)=(bsz,)
            else:
                idx = Categorical(prob_next_node).sample() # size(query)=(bsz,)
            
            # compute logprobs of the action items in the list sumLogProbOfActions   
            ProbOfChoices = prob_next_node[zero_to_bsz, idx] 
            sumLogProbOfActions.append( torch.log(ProbOfChoices) )  # size(query)=(bsz,)

            # update embedding of the current visited node
            h_t = h_encoder[zero_to_bsz, idx, :] # size(h_start)=(bsz, dim_emb)
            h_t = h_t + self.PE[t+1].expand(bsz, self.dim_emb)
            # update tour
            tours.append(idx)

            # update masks with visited nodes
            mask_visited_nodes = mask_visited_nodes.clone()
            mask_visited_nodes[zero_to_bsz, idx] = True
        # logprob_of_choices = sum_t log prob( pi_t | pi_(t-1),...,pi_0 )
        #print('\ntorch.stack(sumLogProbOfActions,dim=1)=',torch.stack(sumLogProbOfActions,dim=1))
        sumLogProbOfActions = torch.stack(sumLogProbOfActions,dim=1).sum(dim=1) # size(sumLogProbOfActions)=(bsz,)

        # convert the list of nodes into a tensor of shape (bsz,num_cities)
        tours = torch.stack(tours,dim=1) # size(col_index)=(bsz, nb_nodes)
        return tours, sumLogProbOfActions
