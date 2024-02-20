import torch 
import numpy as np 
from utils.data_utils import MR_US_dataset
from torch.utils.tensorboard import SummaryWriter
import os 
from utils.data_utils import *
from utils.train_utils import * 
from models.networks import * 
import argparse 

parser = argparse.ArgumentParser(prog='train',
                                description="Script for playing a simple biopsy game")

parser.add_argument('--log_dir',
                    '--log',
                    metavar='log_dir',
                    type=str,
                    action='store',
                    default='pix2pix_augment',
                    help='Log dir to save results to')

parser.add_argument('--num_epochs',
                    metavar='num_epochs',
                    type=str,
                    default='10000',
                    action = 'store',
                    help='How many epochs to play the games for')

parser.add_argument('--batch_size',
                    metavar='batch_size',
                    type=str,
                    default='1',
                    action = 'store',
                    help='Minibatch size')

parser.add_argument('--add_dropout',
                    metavar='add_dropout',
                    type=str,
                    default='False',
                    action = 'store',
                    help='Whether to add drop out or not')

parser.add_argument('--loss_type',
                    metavar='loss_type',
                    type=str,
                    default='combined',
                    action = 'store',
                    help='Which loss function to use : combined (l2+adversarial), l2 only or adversarial only')

parser.add_argument('--l2_weight',
                    metavar='l2_weight',
                    type=str,
                    default='100',
                    action = 'store',
                    help='Coefficient used for l2 loss function in generator')

# Parse arguments
args = parser.parse_args()

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"]="2"
    BASE_FOLDER = 'BASELINE_MODELS'
    SAVE_FOLDER = os.path.join(BASE_FOLDER, args.log_dir)
    BATCH_SIZE = int(args.batch_size)
    NUM_EPOCHS = int(args.num_epochs)
    LAMBDA = int(args.l2_weight)
    LOSS_TYPE = args.loss_type
    
    # Load dataloaders, models 
    data_folder = './evaluate/REGNET'
    train_dataset = MR_US_dataset(data_folder, mode = 'train', alligned = True)
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    val_dataset = MR_US_dataset(data_folder, mode = 'val', alligned = True)
    val_dataloader = DataLoader(val_dataset, batch_size = 1)
    
    # Define models for training 
    discriminator_net = Discriminator(input_channel = 2) # channels being cat layers 
    generator_net = Generator(input_channel = 1) # input is MR image only; output we want is US image 

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Device : {device}")
    
    trainer = Pix2pixTrainer(generator_net, discriminator_net, train_dataloader, val_dataloader, device = device, log_dir = SAVE_FOLDER, loss_type= LOSS_TYPE, l2_weight = LAMBDA)
    trainer.train(NUM_EPOCHS=NUM_EPOCHS)
        
    print(f"Training finished")
    