import os
import torch
import datetime
from utils.options import get_options

# Logs
#os.system("mkdir logs")
args = get_options()
def log():
    print('log begin')
    time_stamp=datetime.datetime.now().strftime("%y%m%d%H%M%S")
    file_name = 'logs'+'/' + "node{}_".format(args.nb_nodes) + time_stamp + ".txt"
    file = open(file_name,"w",1) 
    file.write(time_stamp+'\n\n') 
    for arg in vars(args):
        file.write(arg)
        hyper_param_val="={}".format(getattr(args, arg))
        file.write(hyper_param_val)
        file.write('\n')
    file.write('\n\n') 
    print('log ended')
    return file, time_stamp
    
