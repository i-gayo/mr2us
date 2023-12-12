import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
import os
from torch.utils.data import DataLoader, Dataset
import torch 

class MR_US_dataset(Dataset):
    """
    Dataset that acquires the following: 
    MR
    MR_label
    US
    US_label 
    """
    def __init__(self, dir_name, mode = 'train', downsample = False):
        self.dir_name = dir_name 
        self.mode = mode 
        
        # obtain list of names for us and mri labels 
        self.us_names = os.listdir(os.path.join(dir_name, mode, 'us_images'))
        self.us_label_names = os.listdir(os.path.join(dir_name, mode, 'us_labels'))
        self.mri_names = os.listdir(os.path.join(dir_name, mode, 'mr_images'))
        self.mri_label_names = os.listdir(os.path.join(dir_name, mode, 'mr_labels'))
        self.num_data = len(self.us_names)
        self.downsample = downsample
        
        # Load items 
        self.us_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'us_images', self.us_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        self.us_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode,'us_labels', self.us_label_names[i])).get_fdata()) for i in range(self.num_data)]
        self.mri_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_images', self.mri_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        self.mri_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_labels', self.mri_label_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        
    def __len__(self):
        return self.num_data
        
    def __getitem__(self, idx):
        
        #upsample_us = self.resample(self.us_data[idx])
        #upsample_us_labels = self.resample(self.us_labels[idx], label = True)
        
        # Change view of us image to match mr first 
        if self.alligned:
            # no need to transpose if alligned alerady
            t_us = self.us_data[idx]
            t_us_labels = self.us_labels[idx]
        else:
            t_us = torch.transpose(self.us_data[idx], 2,0)
            t_us_labels = torch.transpose(self.us_labels[idx], 2,0)
            
        # Upsample to MR images 
        upsample_us = self.resample(t_us)
        upsample_us_labels = self.resample(t_us_labels, label = True)
        
        # Add dimesion for "channel"
        mr_data = self.mri_data[idx].unsqueeze(0)
        mr_label = self.mri_labels[idx].unsqueeze(0)
        #mr_label = mr_label[:,:,:,:,0]       # use only prostate label
        
        if len(mr_label.size()) > 4:
            mr_label = mr_label[:,:,:,:,0]       # use only prostate label
        
        # Squeeze dimensions 
        us = upsample_us.squeeze().unsqueeze(0)
        us_label = upsample_us_labels.squeeze().unsqueeze(0)
        
        # normalise data 
        mr = self.normalise_data(mr_data)
        us = self.normalise_data(us)
        mr_label = self.normalise_data(mr_label)
        us_label = self.normalise_data(us_label)
        
        # if resample is true 
        if self.downsample: 
            upsample_method = torch.nn.Upsample(size = (64,64,64))
            mr = upsample_method(mr.unsqueeze(0)).squeeze(0)
            us = upsample_method(us.unsqueeze(0)).squeeze(0)
            mr_label = upsample_method(mr_label.unsqueeze(0)).squeeze(0)
            us_label = upsample_method(us_label.unsqueeze(0)).squeeze(0)
            
        return mr, us, mr_label, us_label
    
    def resample(self, img, dims = (120,128,128), label = False):
        upsample_method = torch.nn.Upsample(size = dims)
        if label: 
            
            if len(img.size()) > 3:
                img_label = img[:,:,:,0]
            else:
                img_label = img 
                
            # Choose only prostate gland label 
            if len(img.size()) > 3:
                img_label = img[:,:,:,0]
            else:
                img_label = img 
                
            upsampled_img = upsample_method(img_label.unsqueeze(0).unsqueeze(0))
        else:
            upsampled_img = upsample_method(img.unsqueeze(0).unsqueeze(0))
        
        return upsampled_img
    
    def normalise_data(self, img):
        """
        Normalises labels and images 
        """
        
        min_val = torch.min(img)
        max_val = torch.max(img)
        
        if max_val == 0: 
            #print(f"Empty mask, not normalised img")
            norm_img = img # return as 0s only if blank image or volume 
        else: 
            norm_img = (img - min_val) / (max_val - min_val)
        
        return norm_img 
    
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