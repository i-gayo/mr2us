import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
import os
from torch.utils.data import DataLoader, Dataset
import torch 

from matplotlib import pyplot as plt
# TODO : Implement dataloaders or use Yipeng's dataloaders 

class MR_US_dataset(Dataset):
    """
    Dataset that acquires the following: 
    MR
    MR_label
    US
    US_label 
    """
    def __init__(self, dir_name, mode = 'train'):
        self.dir_name = dir_name 
        self.mode = mode 
        
        # obtain list of names for us and mri labels 
        self.us_names = os.listdir(os.path.join(dir_name, mode, 'us_images'))
        self.us_label_names = os.listdir(os.path.join(dir_name, mode, 'us_labels'))
        self.mri_names = os.listdir(os.path.join(dir_name, mode, 'mr_images'))
        self.mri_label_names = os.listdir(os.path.join(dir_name, mode, 'mr_labels'))
        self.num_data = len(self.us_names)
        
        # Load items 
        self.us_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'us_images', self.us_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        self.us_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode,'us_labels', self.us_label_names[i])).get_fdata()) for i in range(self.num_data)]
        self.mri_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_images', self.mri_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        self.mri_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_labels', self.mri_label_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        
    def __len__(self):
        return self.num_data
        
    def __getitem__(self, idx):
        
        upsample_us = self.resample(self.us_data[idx]) 
        upsample_us_labels = self.resample(self.us_labels[idx], label = True) 
        return self.mri_data[idx], upsample_us.squeeze().squeeze(), self.mri_labels[idx], upsample_us_labels.squeeze().squeeze()
    
    def resample(self, img, dims = (120,128,128), label = False):
        upsample_method = torch.nn.Upsample(size = dims)
        if label: 
            # Choose only prostate gland label 
            img_label = img[:,:,:,0]
            upsampled_img = upsample_method(img_label.unsqueeze(0).unsqueeze(0))
        else:
            upsampled_img = upsample_method(img.unsqueeze(0).unsqueeze(0))
        
        return upsampled_img
  
if __name__ == '__main__':
    
    data_folder = '/Users/ianijirahmae/Documents/DATASETS/mri_us_paired'
    train_dataset = MR_US_dataset(data_folder, mode = 'train')
    train_dataloader = DataLoader(train_dataset)
    
    for idx, (mr, us, mr_label, us_label) in enumerate(train_dataset):
        print(idx)
        fig, axs = plt.subplots(2,2)
        axs[0,0].imshow(mr[:,:,40])
        axs[0,1].imshow(mr_label[:,:,40,0])
        axs[1,0].imshow(us[:,:,80])
        axs[1,1].imshow(us_label[:,:,80])
        print('chicken')
    
    print('chicken')