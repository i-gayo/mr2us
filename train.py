import torch 
import numpy as np 
from utils.data_utils import MR_US_dataset
from torchmetrics.functional.image import structural_similarity_index_measure as ssim 
from torch.utils.tensorboard import SummaryWriter
import os 
from torch.utils.data import DataLoader, Dataset

class RMSE_loss():
    """
    An RMSE loss function that computes RMSE between each pixels
    """

    def __call__(self, gt, pred):
        """
        Computes RMSE between each individual pixels 
        """
        
        rmse = torch.sqrt(torch.mean((gt - pred)**2))
        
        return rmse 
          
def train_transformnet(model, train_dataset, val_dataset, use_cuda = False, save_folder = 'model'):
    """
    A function that trains a model using dataset 
    """
    
    # Define hyperparameters 
    NUM_EPOCHS = 10000
    LR = 1e-05 
    EVAL_STEPS = 10 # every 10 epochs, compute validation metircs! 
    SAVE_STEPS = 100 #every 100 epochs save new model 
    # Define optimiser and loss functions 
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = RMSE_loss()
    
    # Define tensorboard and saving files 
    os.makedirs(save_folder, exist_ok=True)  #, exists_ok = True)
    writer = SummaryWriter(os.path.join(save_folder, 'runs')) 

    # Save initial losss 
    best_loss = torch.tensor(1000000)

    for epoch in range(NUM_EPOCHS):
        
        print(f"\n Epoch num : {epoch}")
        # Train model 
        model.train()
        
        # Saving lists 
        loss_train = [] 
        ssim_train = [] 
        loss_val = [] 
        ssim_val = [] 
         

        for idx, (mr, us, mr_label, us_label) in enumerate(train_dataset):
            
            # 1. move data and model to gpu 
            if use_cuda: 
                #print(f"Using CUDA")
                mr, us = mr.cuda(), us.cuda()
                model = model.cuda()
                
            # 2. Obtain output of model 
            preds = model(mr.float())
            
            # 3. Compute loss, backpropagate and update weights based on graident 
            optimiser.zero_grad()
            loss = loss_fn(us, preds.float())
            loss.backward()
            optimiser.step()

            # 5. Compute metrics to log (ie loss or other metrics such as MSE or DICE etc)
            with torch.no_grad():
                
                # Compute SSIM 
                ssim_metric = ssim(preds.to(torch.float64).squeeze(1), us)
            
            loss_train.append(loss)
            ssim_train.append(ssim_metric)
        
        # 6. Save metrics to dataloader ; evaluate on validation set every now and then 
        
        # Save to summary writer
        mean_loss = torch.mean(torch.tensor(loss_train))
        mean_ssim = torch.mean(torch.tensor(ssim_train))

        print(f"Epoch {epoch} : mean loss : {mean_loss} ssim : {mean_ssim}")

        writer.add_scalar('Loss/train', mean_loss, epoch)
        writer.add_scalar('SSIM/train', mean_ssim, epoch) 
        
        if (epoch % EVAL_STEPS):
            
            model.eval()
            
            with torch.no_grad():
                
                # Evaluate 
                for idx, (mr, us, mr_label, us_label) in enumerate(val_dataset):
                            
                    if use_cuda: 
                        mr, us = mr.cuda(), us.cuda()
                        
                    preds = model(mr.float())

                    # Compute loss and ssim metrics 
                    loss_eval = loss_fn(us, preds.float())
                    ssim_eval = ssim(preds.to(torch.float64).squeeze(1), us)

                    # Save to val losses 
                    loss_val.append(loss_eval)
                    ssim_val.append(ssim_eval)

                # Save to summary writer
                mean_loss = torch.mean(torch.tensor(loss_val))
                mean_ssim = torch.mean(torch.tensor(ssim_val))
                print(f"Epoch {epoch} : VALIDATION mean loss : {mean_loss} ssim : {mean_ssim}")
                writer.add_scalar('Loss/val', mean_loss, epoch)
                writer.add_scalar('SSIM/val', mean_ssim, epoch) 

                if mean_loss < best_loss:
                    print(f"Saving new model with loss : {mean_loss}")
                    val_path = os.path.join(save_folder, 'best_val_model.pth')
                    torch.save(model.state_dict(), val_path)
                    best_loss = mean_loss 

        if (epoch % SAVE_STEPS):
            train_path = os.path.join(save_folder, 'train_model.pth')
            torch.save(model.state_dict(), train_path)
                                        
if __name__ == '__main__':
    

    from networks import TransformNet
    
    # DEFINE HYPERPARAMETERS
    BATCH_SIZE = 2 
    
    # Define folder names and dataloaders 
    data_folder = './Data'
    train_dataset = MR_US_dataset(data_folder, mode = 'train')
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    val_dataset = MR_US_dataset(data_folder, mode = 'val')
    val_dataloader = DataLoader(val_dataset, batch_size = 1)
    
    # Define model  
    model = TransformNet()
    use_cuda = True
    save_folder = 'BASELINE_v2'
    
    train_transformnet(model, train_dataloader, val_dataloader, use_cuda = True, save_folder = save_folder)
    
    print('chicken')