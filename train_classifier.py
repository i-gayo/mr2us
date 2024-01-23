import torch 
import numpy as np 
from utils.data_utils import MR_US_dataset
from torch.utils.tensorboard import SummaryWriter
import os 
from utils.data_utils import *
from utils.train_utils import * 
from models.networks import * 
import argparse 
CUDA_LAUNCH_BLOCKING = "1"

parser = argparse.ArgumentParser(prog='train',
                                description="Script for training classifier")

parser.add_argument('--log_dir',
                    '--log',
                    metavar='log_dir',
                    type=str,
                    action='store',
                    default='classifier_mr',
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

# Parse arguments
args = parser.parse_args()

if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    SAVE_FOLDER = args.log_dir
    BATCH_SIZE = int(args.batch_size)
    NUM_EPOCHS = int(args.num_epochs)
    EVAL_STEPS = 10 
    SAVE_STEPS = 100 
    BATCH_SIZE = 8 
    
    # Load dataloaders, models 
    #data_folder = './evaluate/REGNET'
    data_folder = './Data'
    train_dataset = MR_dataset(data_folder, mode = 'train', alligned = True)
    train_dataloader = DataLoader(train_dataset, batch_size= BATCH_SIZE)
    val_dataset = MR_dataset(data_folder, mode = 'val', alligned = True)
    val_dataloader = DataLoader(val_dataset, batch_size = 1)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # Define models for training 
    classifier_net = ResNet50(1, 1)
    classifier_net = classifier_net.to(device)
    
    # Define optimisers and loss functions 
    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(classifier_net.parameters())
    
    # Define tensorboard and saving files 
    os.makedirs(SAVE_FOLDER, exist_ok=True)  #, exists_ok = True)
    writer = SummaryWriter(os.path.join(SAVE_FOLDER, 'runs')) 

    # Save initial losss 
    best_loss = torch.tensor(1000000)
    
    for epoch in range(NUM_EPOCHS):
        
        print(f"\n Epoch : {epoch}")
        # Train model 
        classifier_net.train()
        
        # Saving lists 
        loss_train = [] 
        loss_vals = []
        
        for idx, (slice, whole_slice, label) in enumerate(train_dataloader):
            
            slice = slice.to(device)
            label = label.to(device)
            
            # Compute predictions 
            pred_label = classifier_net(slice.float())
            
            # Compute loss 
            opt.zero_grad() # zero gradients
            #print(pred_label)
            #print(label)
            loss_val = loss_fn(pred_label, label.unsqueeze(1))
            loss_val.backward()
            loss_train.append(loss_val.item())
            
            # Adjust weights
            opt.step()
            
            #print('fuecoco')
        
        # Save results 
        with torch.no_grad():
            mean_loss = torch.mean(torch.tensor(loss_train))
            
        print(f"Epoch : {epoch} mean loss : {mean_loss}")
        writer.add_scalar('Loss/train', mean_loss, epoch)
        
        if (epoch % EVAL_STEPS):
            
            classifier_net.eval()
            
            with torch.no_grad():
                
                # Evaluate 
                for idx, (us, us_slice, us_label) in enumerate(val_dataloader):
                        
                    us = us.to(device)
                    us_label = us_label.to(device)
                    
                    preds = classifier_net(us.float())
                    loss_eval = loss_fn(preds, us_label.unsqueeze(1))
                    
                    # Save to val losses 
                    loss_vals.append(loss_eval)

                # Save to summary writer
                mean_loss = torch.mean(torch.tensor(loss_val))
                print(f"Epoch {epoch} : VALSAVEIDATION mean loss : {mean_loss}")
                writer.add_scalar('Loss/val', mean_loss, epoch)

                if mean_loss < best_loss:
                    print(f"Saving new model with loss : {mean_loss}")
                    val_path = os.path.join(SAVE_FOLDER, 'best_val_model.pth')
                    torch.save(classifier_net.state_dict(), val_path)
                    best_loss = mean_loss 

        if (epoch % SAVE_STEPS):
            print(f"Saving model periodically : epoch {epoch}")
            train_path = os.path.join(SAVE_FOLDER, 'train_model.pth')
            torch.save(classifier_net.state_dict(), train_path)
                             
    
    print('fuecoco')