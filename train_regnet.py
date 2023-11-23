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
                    default='pix2pix',
                    help='Log dir to save results to')

parser.add_argument('--num_epochs',
                    metavar='num_epochs',
                    type=str,
                    default='1000',
                    action = 'store',
                    help='How many epochs to play the games for')

parser.add_argument('--batch_size',
                    metavar='--batch_size',
                    type=str,
                    default='2',
                    action = 'store',
                    help='Minibatch size')

# Parse arguments
args = parser.parse_args()

if __name__ == "__main__":
    

    BATCH_SIZE = 2
    SAVE_FOLDER = args.log_dir
    BATCH_SIZE = int(args.batch_size)
    NUM_EPOCHS = int(args.num_epochs)
    
    # Load dataloaders, models 
    data_folder = './Data'
    train_dataset = MR_US_dataset(data_folder, mode = 'train')
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    val_dataset = MR_US_dataset(data_folder, mode = 'val')
    val_dataloader = DataLoader(val_dataset, batch_size = 1)
    
    # obtain shape of input data
    mr, us, mr_label, us_label = train_dataset[0]
    mr_us = torch.cat((mr.unsqueeze(0),us.unsqueeze(0)), dim = 1)
    input_shape = (mr_us.size()[-3:])
    
    # Obtain device 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Device : {device}")
    
    # Set up model and trainer 
    reg_model = LocalNet(input_shape, device = device, input_channel = 2)
    reg_trainer = LocalNetTrainer(reg_model, train_dataloader, val_dataloader, device = device, log_dir = 'regnet')
    
    reg_trainer.train()
    
    print('chicken')
    