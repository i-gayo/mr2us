import torch 
import numpy as np 
from data_utils import MR_US_dataset
from networks import TransformNet

def rmse_loss(gt, preds):
    """
    TODO: Computes RMSE from ground truth to pred image 
    """
    
    raise NotImplemented

def train_script(dataset, model, cuda = 'cuda'):
    """
    A function that trains a model using dataset 
    """
    
    for idx, (mr, us, mr_label, us_label) in enumerate(dataset):
        
        # 1. move data and model to gpu 
        pass 
    
        # 2. Obtain output of model 
        preds = model(mr)
        
        # 3. Compute loss function 
        loss = rmse_loss(us, preds)
        
        # 4. Backpropagate loss 
        
        
        # 5. Compute metrics to log (ie loss or other metrics such as MSE or DICE etc)
        
    pass 


if __name__ == '__main__':
    
    