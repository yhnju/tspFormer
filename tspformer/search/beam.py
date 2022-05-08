import torch
import torch.nn as nn
from utils.options import get_options
from torch.distributions.categorical import Categorical
from utils.positionEncode import generate_positional_encoding

if torch.cuda.is_available():
    device = torch.device("cuda")

def beam0(h_encoder,decoder,x,B,t,K_att_decoder,V_att_decoder,K_att_decoder_tmp,V_att_decoder_tmp):
    # some parameters
    bsz = x.shape[0]
    nb_nodes = x.shape[1]
    zero_to_bsz = torch.arange(bsz, device=x.device) # [0,1,...,bsz-1]
         # For beam search
    zero_to_B = torch.arange(B, device=x.device) # [0,1,...,B-1]

    args = get_options()
    PE = generate_positional_encoding(args.dim_emb, args.max_len_PE).to(x.device)        

    B_t0 = min(B, nb_nodes)
    # input placeholder that starts the decoding
    idx_start_placeholder = torch.Tensor([nb_nodes]).long().repeat(bsz).to(x.device)
    h_start = h_encoder[zero_to_bsz, idx_start_placeholder, :] + PE[0].repeat(bsz,1) # size(h_start)=(bsz, dim_emb)
    h_t = h_start # size(h_start)=(bsz, dim_emb)
    mask_visited_nodes = torch.zeros(bsz, nb_nodes+1, device=x.device).bool() # False, size(mask_visited_nodes)=(bsz, nb_nodes+1) # initialize mask of visited cities
    mask_visited_nodes[zero_to_bsz, idx_start_placeholder] = True
    # compute probability over the next node in the tour
    prob_next_node = decoder(h_t, K_att_decoder, V_att_decoder, mask_visited_nodes) # size(prob_next_node)=(bsz, nb_nodes+1) 
    # compute score_t + sum_t score_{t-1} for all beams
    score_t = torch.log(prob_next_node) # size(score_t)=(bsz, nb_nodes+1) for t=0 
    sum_scores = score_t # size(score_t)=(bsz, nb_nodes+1)
    # choose nodes with top-B sumScores 
    #print('sum_scores=',sum_scores)
    top_val, top_idx = torch.topk(sum_scores, B_t0, dim=1) # size(sumScores)=(bsz, B_t0)
    # update sum_t score_{t} for all beams
    #print('top_val=',top_val,'\ntop_idx=',top_idx)
    sum_scores = top_val # size(sumScores)=(bsz, B_t0) 
    zero_to_B_t0 = torch.arange(B_t0, device=x.device) # [0,1,...,B_t0-1]
    mask_visited_nodes = mask_visited_nodes.unsqueeze(1) # size(mask_visited_nodes)=(bsz, 1, nb_nodes+1)
    mask_visited_nodes = torch.repeat_interleave(mask_visited_nodes, B_t0, dim=1)
    for b in range(bsz):
        mask_visited_nodes[b, zero_to_B_t0, top_idx[b]] = True # size(mask_visited_nodes)=(bsz, B_t0, nb_nodes+1)

    tours = torch.zeros(bsz, B_t0, nb_nodes, device=x.device).long() # size(tours)=(bsz, B_t0, nb_nodes)
    tours[:,:,t] = top_idx # size(tours)=(bsz, B_t0, nb_nodes)
        # update embedding of the current visited node
    h_t = torch.zeros(bsz, B_t0, args.dim_emb, device=x.device) # size(tours)=(bsz, B_t0, dim_emb)
    for b in range(bsz):
        h_t[b, :, :] = h_encoder[b, top_idx[b], :] # size(h_t)=(bsz, B_t0, dim_emb)
    h_t = h_t + PE[t+1].expand(bsz, B_t0, args.dim_emb) # size(h_t)=(bsz, B_t0, dim_emb)
    decoder.repeat_selfatt_keys_values(B_t0)
    K_att_decoder = torch.repeat_interleave(K_att_decoder_tmp, B_t0, dim=0) 
    # size(K_att_decoder)=(bsz*B_t0, nb_nodes+1, dim_emb*nb_layers_decoder)
    V_att_decoder = torch.repeat_interleave(V_att_decoder_tmp, B_t0, dim=0) 
    # size(V_att_decoder)=(bsz*B_t0, nb_nodes+1, dim_emb*nb_layers_decoder)
    return K_att_decoder, V_att_decoder, h_t, mask_visited_nodes, sum_scores,tours


def beam1(h_encoder,decoder,x,B,t,h_t,mask_visited_nodes,K_att_decoder, V_att_decoder 
    ,sum_scores,tours,K_att_decoder_tmp,V_att_decoder_tmp):
    # some parameters
    print('\nt=1,beam1 begins.')
    bsz = x.shape[0]
    nb_nodes = x.shape[1]
    zero_to_bsz = torch.arange(bsz, device=x.device) # [0,1,...,bsz-1]
         # For beam search
    zero_to_B = torch.arange(B, device=x.device) # [0,1,...,B-1]

    args = get_options()
    PE = generate_positional_encoding(args.dim_emb, args.max_len_PE).to(x.device)        

    B_t0 = min(B, nb_nodes)

    # compute probability over the next node in the tour
    h_t = h_t.view(bsz*B_t0, args.dim_emb)
    mask_visited_nodes = mask_visited_nodes.view(bsz*B_t0, nb_nodes+1)
    prob_next_node = decoder(h_t, K_att_decoder, V_att_decoder, mask_visited_nodes) # size(prob_next_node)=(bsz.B_t0, nb_nodes+1) 
    prob_next_node = prob_next_node.view(bsz, B_t0, nb_nodes+1) # size(prob_next_node)=(bsz, B_t0, nb_nodes+1) 
    mask_visited_nodes = mask_visited_nodes.view(bsz, B_t0, nb_nodes+1)
    h_t = h_t.view(bsz, B_t0, args.dim_emb) 
    # compute score_t + sum_t score_{t-1} for all beams
    score_t = torch.log(prob_next_node) # size(score_t)=(bsz, B, nb_nodes+1) 
    sum_scores = score_t + sum_scores.unsqueeze(2) # size(score_t)=(bsz, B, nb_nodes+1)
    sum_scores_flatten = sum_scores.view(bsz, -1) # size(sumScores_next_node)=(bsz, B.(nb_nodes+1))
    # choose nodes with top-B sumScores 
    top_val, top_idx = torch.topk(sum_scores_flatten, B, dim=1)
    idx_top_beams = top_idx // (nb_nodes+1) # size(idx_beam_topB)=(bsz, B)
    idx_in_beams = top_idx - idx_top_beams* (nb_nodes+1) # size(idx_in_beams)=(bsz, B)
    # update sum_t score_{t} for all beams
    sum_scores = top_val
    # update beam masks with visited nodes
    mask_visited_nodes_tmp = mask_visited_nodes.clone() # size(mask_visited_nodes_tmp)=(bsz, B_t0, nb_nodes+1)
    mask_visited_nodes = torch.zeros(bsz, B, nb_nodes+1, device=x.device).bool() # size(mask_visited_nodes)=(bsz, B, nb_nodes+1)
    for b in range(bsz):
        mask_visited_nodes[b, zero_to_B, :] = mask_visited_nodes_tmp[b, idx_top_beams[b], :] # size(mask_visited_nodes)=(bsz, B, nb_nodes+1)
    for b in range(bsz):
        mask_visited_nodes[b, zero_to_B, idx_in_beams[b]] = True # size(mask_visited_nodes)=(bsz, B, nb_nodes+1)
           # update beam tours with visited nodes
    tours_tmp = tours.clone()
    tours = torch.zeros(bsz, B, nb_nodes, device=x.device).long() # size(tours)=(bsz, B, nb_nodes)
    for b in range(bsz):
        tours[b, zero_to_B, :] = tours_tmp[b, idx_top_beams[b], :]
    tours[:,:,t] = idx_in_beams # size(tours)=(bsz, B, nb_nodes)
           # update embedding of the current visited node
    h_t = torch.zeros(bsz, B, args.dim_emb, device=x.device) # size(tours)=(bsz, B_t0, dim_emb)
    for b in range(bsz):
        h_t[b, :, :] = h_encoder[b, idx_in_beams[b], :] # size(h_t)=(bsz, B, dim_emb)
    h_t = h_t + PE[t+1].expand(bsz, B, args.dim_emb) # size(h_t)=(bsz, B, dim_emb)
          # update self-attention embeddings of partial tours
    decoder.reorder_selfatt_keys_values(t, idx_top_beams) # size(K_att_decoder)=(bsz*B_t0, nb_nodes+1, dim_emb*nb_layers_decoder)
    
    K_att_decoder = torch.repeat_interleave(K_att_decoder_tmp, B, dim=0) # size(K_att_decoder)=(bsz*B, nb_nodes+1, dim_emb*nb_layers_decoder)
    V_att_decoder = torch.repeat_interleave(V_att_decoder_tmp, B, dim=0) # size(V_att_decoder)=(bsz*B, nb_nodes+1, dim_emb*nb_layers_decoder)
    return K_att_decoder, V_att_decoder,h_t, mask_visited_nodes, sum_scores, tours

def beamGreaterEq2(h_encoder,decoder,x,B,t,h_t,mask_visited_nodes,K_att_decoder, V_att_decoder, sum_scores, tours):
    # some parameters
    #K_att_decoder_tmp=None; V_att_decoder_tmp=None
    '''K_att_decoder, V_att_decoder,h_t, mask_visited_nodes, sum_scores, tours = \
                    beam1(h_encoder,decoder,x,B,t,h_t,mask_visited_nodes,K_att_decoder,
                     V_att_decoder,sum_scores,tours, None,None)'''
    bsz = x.shape[0]
    nb_nodes = x.shape[1]
    zero_to_bsz = torch.arange(bsz, device=x.device) # [0,1,...,bsz-1]
         # For beam search
    zero_to_B = torch.arange(B, device=x.device) # [0,1,...,B-1]

    args = get_options()
    PE = generate_positional_encoding(args.dim_emb, args.max_len_PE).to(x.device)        
 
    B_t0 = min(B, nb_nodes)

               # compute probability over the next node in the tour
    h_t = h_t.view(bsz*B, args.dim_emb)
    mask_visited_nodes = mask_visited_nodes.view(bsz*B, nb_nodes+1)
    prob_next_node = decoder(h_t, K_att_decoder, V_att_decoder, mask_visited_nodes) # size(prob_next_node)=(bsz.B, nb_nodes+1) 
    prob_next_node = prob_next_node.view(bsz, B, nb_nodes+1) # size(prob_next_node)=(bsz, B, nb_nodes+1) 
    mask_visited_nodes = mask_visited_nodes.view(bsz, B, nb_nodes+1)
    h_t = h_t.view(bsz, B, args.dim_emb) 
            # compute score_t + sum_t score_{t-1} for all beams
    score_t = torch.log(prob_next_node) # size(score_t)=(bsz, B, nb_nodes+1)
    sum_scores = score_t + sum_scores.unsqueeze(2) # size(score_t)=(bsz, B, nb_nodes+1)
    sum_scores_flatten = sum_scores.view(bsz, -1) # size(sumScores_next_node)=(bsz, B.(nb_nodes+1))
          # choose nodes with top-B sumScores 
    top_val, top_idx = torch.topk(sum_scores_flatten, B, dim=1)
    idx_top_beams = top_idx // (nb_nodes+1) # size(idx_beam_topB)=(bsz, B)
    idx_in_beams = top_idx - idx_top_beams* (nb_nodes+1) # size(idx_in_beams)=(bsz, B)
        # update sum_t score_{t} for all beams
    sum_scores = top_val
       # update beam masks with visited nodes
    mask_visited_nodes_tmp = mask_visited_nodes.clone()
    for b in range(bsz):
        mask_visited_nodes[b, zero_to_B, :] = mask_visited_nodes_tmp[b, idx_top_beams[b], :]
    for b in range(bsz):
        mask_visited_nodes[b, zero_to_B, idx_in_beams[b]] = True
         # update beam tours with visited nodes
    tours_tmp = tours.clone()
    for b in range(bsz):
        tours[b, zero_to_B, :] = tours_tmp[b, idx_top_beams[b], :]
    tours[:,:,t] = idx_in_beams # size(tours)=(bsz, B, nb_nodes)
        # update embedding of the current visited node
    for b in range(bsz):
        h_t[b, :, :] = h_encoder[b, idx_in_beams[b], :] # size(h_t)=(bsz, B, dim_emb)
    h_t = h_t + PE[t+1].expand(bsz, B, args.dim_emb) # size(h_t)=(bsz, B, dim_emb)
        # update self-attention embeddings of partial tours
    decoder.reorder_selfatt_keys_values(t, idx_top_beams)
        # sum_t log prob( pi_t | pi_0,...pi_(t-1) )
    sum_scores = sum_scores[:,0] # size(sumScores)=(bsz)
    tours_beamsearch = tours
    scores_beamsearch = sum_scores
    return tours_beamsearch, scores_beamsearch
