import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
import os
from torch.utils.data import DataLoader, Dataset

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
        self.us_data = [nib.load(os.path.join(dir_name, mode, 'us_images', self.us_names [i])).squeeze().get_fdata() for i in range(self.num_data)]
        self.us_labels = [nib.load(os.path.join(dir_name, mode,'us_labels', self.us_label_names[i])).get_fdata() for i in range(self.num_data)]
        self.mri_data = [nib.load(os.path.join(dir_name, mode, 'mr_images', self.mri_names [i])).get_fdata() for i in range(self.num_data)]
        self.mri_labels = [nib.load(os.path.join(dir_name, mode, 'mr_labels', self.mri_label_names [i])).get_fdata() for i in range(self.num_data)]
        
    def __len__(self):
        return self.num_data
        
    def __getitem__(self, idx):
        #test = self.resample(0, 0)
        
        return self.mri_data[idx], self.us_data[idx], self.mri_labels[idx], self.us_labels[idx]
    
    def resample(self, img,label, dims = np.array([120, 128, 18])):
        raise NotImplemented
        
        
    
    
if __name__ == '__main__':
    
    data_folder = '/Users/ianijirahmae/Documents/DATASETS/mri_us_paired'
    train_dataset = MR_US_dataset(data_folder, mode = 'train')
    train_dataloader = DataLoader(train_dataset)
    
    for idx, (mr, us, mr_label, us_label) in enumerate(train_dataset):
        print(idx)
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(mr[:,:,40,0])
        axs[1].imshow(np.max(mr_label[:,:,:,0], axis =2))
        print('chicken')
    print('chicken')