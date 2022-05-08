import torch
import torch.nn as nn
from utils.options import get_options
from torch.distributions.categorical import Categorical
from utils.positionEncode import generate_positional_encoding

if torch.cuda.is_available():
    device = torch.device("cuda")

    # Greedy search
def greedySearch(h_encoder,decoder,x,B):
    #print('Greedy decoding')
    # some parameters
    bsz = x.shape[0]
    nb_nodes = x.shape[1]
    zero_to_bsz = torch.arange(bsz, device=x.device) # [0,1,...,bsz-1]
    args = get_options()


    PE = generate_positional_encoding(args.dim_emb, args.max_len_PE).to(x.device)        
        
    deterministic = True
    # list that will contain Long tensors of shape (bsz,) that gives the idx of the cities chosen at time t
    tours = []
    # list that will contain Float tensors of shape (bsz,) that gives the neg log probs of the choices made at time t
    sumLogProbOfActions = []
    # input placeholder that starts the decoding
    idx_start_placeholder = torch.Tensor([nb_nodes]).long().repeat(bsz).to(x.device)
    h_start = h_encoder[zero_to_bsz, idx_start_placeholder, :] + PE[0].repeat(bsz,1) # size(h_start)=(bsz, dim_emb)
    # initialize mask of visited cities
    mask_visited_nodes = torch.zeros(bsz, nb_nodes+1, device=x.device).bool() # False
    mask_visited_nodes[zero_to_bsz, idx_start_placeholder] = True
    # clear key and val stored in the decoder
    decoder.reset_selfatt_keys_values()

        # key and value for decoder    
    WK_att_decoder = nn.Linear(args.dim_emb, args.nb_layers_decoder* args.dim_emb) 
    WV_att_decoder = nn.Linear(args.dim_emb, args.nb_layers_decoder* args.dim_emb) 
    WK_att_decoder = WK_att_decoder.to(device)
    WV_att_decoder = WV_att_decoder.to(device)

    K_att_decoder = WK_att_decoder(h_encoder) # size(K_att)=(bsz, nb_nodes+1, dim_emb*nb_layers_decoder)
    V_att_decoder = WV_att_decoder(h_encoder) # size(V_att)=(bsz, nb_nodes+1, dim_emb*nb_layers_decoder)

    # construct tour recursively
    h_t = h_start
    for t in range(nb_nodes):
        # compute probability over the next node in the tour
        prob_next_node = decoder(h_t, K_att_decoder, V_att_decoder, mask_visited_nodes) # size(prob_next_node)=(bsz, nb_nodes+1)
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
        h_t = h_t + PE[t+1].expand(bsz, args.dim_emb)
        # update tour
        tours.append(idx)
        # update masks with visited nodes
        mask_visited_nodes = mask_visited_nodes.clone()
        mask_visited_nodes[zero_to_bsz, idx] = True
    # logprob_of_choices = sum_t log prob( pi_t | pi_(t-1),...,pi_0 )
    sumLogProbOfActions = torch.stack(sumLogProbOfActions,dim=1).sum(dim=1) # size(sumLogProbOfActions)=(bsz,)
    # convert the list of nodes into a tensor of shape (bsz,num_cities)
    tours = torch.stack(tours,dim=1) # size(col_index)=(bsz, nb_nodes)
    tours_greedy = tours
    scores_greedy = sumLogProbOfActions 
    return tours_greedy, scores_greedy