import os
import time
import torch
import argparse

def get_options(args=None):
    parser = argparse.ArgumentParser(description="Tspformer")

    # Data
    parser.add_argument('--nb_nodes', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--dim_input_nodes', type=int, default=2, help='Dimension of city location,(x, y)')
    parser.add_argument('--gpu_id', type=str, default='0', help='The index of GPU')
    parser.add_argument('--batchnorm', action='store_true', default=False, help='batch or layer normilizaiotn')
    parser.add_argument('--max_len_PE', type=int, default=1000, help='Number of positonal encoding')

    # Model
    parser.add_argument('--dim_emb', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--dim_ff', type=int, default=512, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--nb_layers_encoder', type=int, default=6,help='Number of layers in the encoder')
    parser.add_argument('--nb_layers_decoder', type=int, default=2,help='Number of layers in the decoder')

    # Training
    parser.add_argument('--lr', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--tol', type=float, default=1e-3, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch')
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--nb_epochs', type=int, default=10000, help='The number of epochs to train')
    parser.add_argument('--nb_batch_per_epoch', type=int, default=3000, help='The number of iteratons per epochs ')
    parser.add_argument('--nb_batch_eval', type=int, default=100, help='The number of evaluation ')

    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay (default 0.8)')
    parser.add_argument('--baseline',type=str, default='rollout',
                        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                        help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
                             'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
    parser.add_argument('--eval_batch_size', type=int, default=100,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                             ' to save memory (default None means no shrinking)')
    parser.add_argument('--data_distribution', type=str, default=None,
                        help='Data distribution to use during training, defaults and options depend on problem.')

    # Misc
    parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
    parser.add_argument('--log_dir',type=str, default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name',type=str, default='yang', help='Name to identify the run')
    parser.add_argument('--output_dir',type=str, default='outputs', help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=5,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--load_path',type=str, help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume',type=str, help='Resume from previous checkpoint file')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    opts = parser.parse_args(args)

    return opts
