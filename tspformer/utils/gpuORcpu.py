import os
import torch

def gpuORcpu():
    device = torch.device("cpu") # select CPU
    gpu_id = '0' # select a single GPU , gpu_id = '2,3', select multiple GPUs  
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # select gpu id you want to use,such as '0,1,2,3'
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('Using GPU training.GPU numbers: {:d}, GPU name: {:s}, gpu_id: {:s}'.format(torch.cuda.device_count(),
        torch.cuda.get_device_name(0), gpu_id))   
    else:
        print('Using CPU training.')
    return device