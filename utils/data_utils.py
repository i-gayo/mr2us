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
    
    def __init__(self, dir_name, mode = 'train', downsample = False, alligned = False, get_2d = False):
        self.dir_name = dir_name 
        self.mode = mode 
        
        # obtain list of names for us and mri labels 
        self.us_names = os.listdir(os.path.join(dir_name, mode, 'us_images'))
        self.us_label_names = os.listdir(os.path.join(dir_name, mode, 'us_labels'))
        self.mri_names = os.listdir(os.path.join(dir_name, mode, 'mr_images'))
        self.mri_label_names = os.listdir(os.path.join(dir_name, mode, 'mr_labels'))
        self.num_data = len(self.us_names)
        self.downsample = downsample
        self.alligned = alligned
        self.get_2d = get_2d # whether to get whole volume or 2d slices only
        
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
        
        if self.alligned:
            # no need to transpose if alligned alerady
            t_us = self.us_data[idx]
            t_us_labels = self.us_labels[idx]
        else: # Change view of us image to match mr first 
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
        
        if self.get_2d:
            # Returns only slice ie 1 x width x height only (axial direction obtains)
            num_slices = mr.size()[-1]
            # TODO ; option to obtain inner slices only
            slice_idx = np.random.choice(np.arange(0,num_slices-1))
            mr = mr[:,:,:,slice_idx]     
            mr_label = mr_label[:,:,:,slice_idx]
            us = us[:,:,:,slice_idx]
            us_label = us_label[:,:,:,slice_idx]
               
        return mr, us, mr_label, us_label
    
    def resample(self, img, dims = (120,128,128), label = False):
        upsample_method = torch.nn.Upsample(size = dims)
        if label: 
            
            if len(img.size()) > 4:
                img_label = img[:,:,:,0]
            else:
                img_label = img 
                
            # # Choose only prostate gland label 
            # if len(img.size()) > 3:
            #     img_label = img[:,:,:,0]
            # else:
            #     img_label = img 
            
            if len(img.size()) == 4:
                img_to_upsample = img_label.unsqueeze(0)
            else:
                img_to_upsample = img_label.unsqueeze(0).unsqueeze(0)
            upsampled_img = upsample_method(img_to_upsample)
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
    
class US_dataset(Dataset):
    
    def __init__(self, dir_name, mode = 'train', downsample = False, alligned = False, sample_fair = False, give_sim = False):
        self.dir_name = dir_name 
        self.mode = mode 
        self.sample_fair = sample_fair 
        # obtain list of names for us and mri labels 
        self.us_names = os.listdir(os.path.join(dir_name, mode, 'us_images'))
        self.us_label_names = os.listdir(os.path.join(dir_name, mode, 'us_labels'))
        
        # self.mri_names = os.listdir(os.path.join(dir_name, mode, 'mr_images'))
        # self.mri_label_names = os.listdir(os.path.join(dir_name, mode, 'mr_labels'))
        self.num_data = len(self.us_names)
        self.downsample = downsample
        self.alligned = alligned
        
        # Load items 
        self.us_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'us_images', self.us_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        self.us_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode,'us_labels', self.us_label_names[i])).get_fdata()) for i in range(self.num_data)]
        #self.mri_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_images', self.mri_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        #self.mri_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_labels', self.mri_label_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        
    def __len__(self):
        return self.num_data
        
    def __getitem__(self, idx):
        
        #upsample_us = self.resample(self.us_data[idx])
        #upsample_us_labels = self.resample(self.us_labels[idx], label = True)
        
        if self.alligned:
            # no need to transpose if alligned alerady
            t_us = self.us_data[idx]
            t_us_labels = self.us_labels[idx]
        else: # Change view of us image to match mr first 
            
            # dimensions are 88 x 118 x 81 x 6 for US labels 
            t_us = torch.transpose(self.us_data[idx], 2,0)
            t_us_labels = torch.transpose(self.us_labels[idx], 2,0)
            
        # Upsample to MR images 
        upsample_us = self.resample(t_us)
        upsample_us_labels = self.resample(t_us_labels, label = True)
        
        # Squeeze dimensions 
        us = upsample_us.squeeze().unsqueeze(0)
        us_label = upsample_us_labels.squeeze().unsqueeze(0)
        
        # normalise data 
        #mr = self.normalise_data(mr_data)
        #mr_label = self.normalise_data(mr_label)
        us = self.normalise_data(us)
        us_label = self.normalise_data(us_label)
        
        # if resample is true 
        if self.downsample: 
            upsample_method = torch.nn.Upsample(size = (64,64,64))
            #mr = upsample_method(mr.unsqueeze(0)).squeeze(0)
            #mr_label = upsample_method(mr_label.unsqueeze(0)).squeeze(0)
            us = upsample_method(us.unsqueeze(0)).squeeze(0)
            us_label = upsample_method(us_label.unsqueeze(0)).squeeze(0)
        
        # Sample a slice and corresponding label 
        us_shape = us.size()
        #WITH_PROSTATE = [idx for idx in range(us_shape[-1]) if torch.any(us_label[:,:,idx] > 0)]
        # randomly sample US slice 
        if self.sample_fair:
            WITH_PROSTATE = [idx for idx in range(us_shape[-1]) if torch.any(us_label[:,:,idx] > 0)]
            WITHOUT_PROSTATE = [idx for idx in range(us_shape[-1]) if not(torch.any(us_label[:,:,idx] > 0))]
            # Find index of slices WITH prostate 
            p = np.random.rand()
            # Find index of slices WITHOUT prostate
            if p < 0.5:
                slice_idx = np.random.choice(WITH_PROSTATE)
            else:
                slice_idx = np.random.choice(WITHOUT_PROSTATE)
            # if p < 0.5 : with prostate
            
            # if p > 0.5 : without prostate 
        else:
            slice_idx = np.random.choice(np.arange(0,us_shape[-1]))
        
        #print(f"Slice idx {slice_idx}")
        # us is 1 x width x ehight x depth; so obtaining slingle slice
        us_slice = us[:,:,:,slice_idx] # changed from [:,:,slice_idx]
        us_label_slice = us_label[:,:,:,slice_idx] # segmented slice 
        us_label_classify = torch.tensor(1.0*(len(torch.unique(us_label_slice)) > 1)) # 1 if presence of prostate 
        
        return us_slice, us_label_slice, us_label_classify
    
    def resample(self, img, dims = (120,128,128), label = False):
        upsample_method = torch.nn.Upsample(size = dims)
        if label: 
            
            # Us size : 88 x 118 x 81 x 6 
            if self.alligned == False:
                img = img[:,:,:,0] # use only prostate label 
            
            # if len(img.size()) > 3:
            #     img_label = img[:,:,:,0]
            # else:
            #     img_label = img 
                
            # Choose only prostate gland label 
            if len(img.size()) == 4:
                img_label = img.unsqueeze(0)
            else:
                img_label = img.unsqueeze(0).unsqueeze(0)
            
            # In the size : 1 x 1 x width x height x depth
            upsampled_img = upsample_method(img_label)
            #upsampled_img = upsample_method(img_to_upsample)
            #upsampled_img = upsample_method(img_label.unsqueeze(0).unsqueeze(0))
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

class MR_dataset(Dataset):
    
    def __init__(self, dir_name, mode = 'train', downsample = False, alligned = False, sample_fair = False):
        self.dir_name = dir_name 
        self.mode = mode 
        self.sample_fair = sample_fair
        
        # obtain list of names for us and mri labels 
        #self.us_names = os.listdir(os.path.join(dir_name, mode, 'us_images'))
        #self.us_label_names = os.listdir(os.path.join(dir_name, mode, 'us_labels'))
        
        self.mri_names = os.listdir(os.path.join(dir_name, mode, 'mr_images'))
        self.mri_label_names = os.listdir(os.path.join(dir_name, mode, 'mr_labels'))
        self.num_data = len(self.mri_names)
        self.downsample = downsample
        self.alligned = alligned
        
        # Load items 
        #self.us_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'us_images', self.us_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        #self.us_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode,'us_labels', self.us_label_names[i])).get_fdata()) for i in range(self.num_data)]
        self.mri_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_images', self.mri_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        self.mri_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_labels', self.mri_label_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        
    def __len__(self):
        return self.num_data
        
    def __getitem__(self, idx):
        
        #upsample_us = self.resample(self.us_data[idx])
        #upsample_us_labels = self.resample(self.us_labels[idx], label = True)
        
        # if self.alligned:
        #     # no need to transpose if alligned alerady
        #     t_us = self.us_data[idx]
        #     t_us_labels = self.us_labels[idx]
        # else: # Change view of us image to match mr first 
        #     t_us = torch.transpose(self.us_data[idx], 2,0)
        #     t_us_labels = torch.transpose(self.us_labels[idx], 2,0)
            
        # Upsample to MR images 
        #upsample_us = self.resample(t_us)
        #upsample_us_labels = self.resample(t_us_labels, label = True)
        
        # Squeeze dimensions 
        #us = upsample_us.squeeze().unsqueeze(0)
        #us_label = upsample_us_labels.squeeze().unsqueeze(0)
        
        # Add dimesion for "channel"
        mr_data = self.mri_data[idx].unsqueeze(0)
        mr_label = self.mri_labels[idx].unsqueeze(0)
        
        # normalise data 
        mr = self.normalise_data(mr_data)
        mr_label = self.normalise_data(mr_label)
        #us = self.normalise_data(us)
        #us_label = self.normalise_data(us_label)
        
        # if resample is true 
        if self.downsample: 
            upsample_method = torch.nn.Upsample(size = (64,64,64))
            mr = upsample_method(mr.unsqueeze(0)).squeeze(0)
            mr_label = upsample_method(mr_label.unsqueeze(0)).squeeze(0)
            #us = upsample_method(us.unsqueeze(0)).squeeze(0)
            #us_label = upsample_method(us_label.unsqueeze(0)).squeeze(0)
        
        # Sample a slice and corresponding label 
        #us_shape = us.size()
        mr_shape = mr.size()
        
        if self.sample_fair:
            WITH_PROSTATE = [idx for idx in range(mr_shape[-1]) if torch.any(mr_label[:,:,idx] > 0)]
            WITHOUT_PROSTATE = [idx for idx in range(mr_shape[-1]) if not(torch.any(mr_label[:,:,idx] > 0))]
            # Find index of slices WITH prostate 
            p = np.random.rand()
            # Find index of slices WITHOUT prostate
            if p < 0.5:
                print(f"Sampled with prostate")
                slice_idx = np.random.choice(WITH_PROSTATE)
            else:
                print(f"Sampled without prostate")
                slice_idx = np.random.choice(WITHOUT_PROSTATE)
            # if p < 0.5 : with prostate
            
            # if p > 0.5 : without prostate 
            
            mr_slice = mr[:,:,:,slice_idx]
            mr_label_slice = mr_label[:,:,:,slice_idx] # segmented slice 
            mr_label_classify = torch.tensor(1.0*(len(torch.unique(mr_label_slice)) > 1)) # 1 if presence of prostate 

            # RETURN slice with 50% chance with prostate, 50% no prostate 
            return mr_slice, mr_label_slice, mr_label_classify
    
        else:
            #slice_idx = np.random.choice(np.arange(0,mr_shape[-1]))
            # Else : retutrn entire volume 
            
            return mr, mr_label
        

    
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

class US_dataset_withfake(Dataset):
    
    def __init__(self, dir_name, mode = 'train', downsample = False, alligned = False, sample_fair = False):
        self.dir_name = dir_name 
        self.mode = mode 
        self.sample_fair = sample_fair

        # obtain list of names for us and mri labels 
        self.us_names = os.listdir(os.path.join(dir_name, mode, 'us_images'))
        self.us_label_names = os.listdir(os.path.join(dir_name, mode, 'us_labels'))
        
        # self.mri_names = os.listdir(os.path.join(dir_name, mode, 'mr_images'))
        # self.mri_label_names = os.listdir(os.path.join(dir_name, mode, 'mr_labels'))
        self.num_data = len(self.us_names)
        self.downsample = downsample
        self.alligned = alligned
        
        # Load items 
        self.us_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'us_images', self.us_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        self.us_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode,'us_labels', self.us_label_names[i])).get_fdata()) for i in range(self.num_data)]
        self.fake_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'fake_us_images', self.us_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        
        #self.mri_data = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_images', self.mri_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        #self.mri_labels = [torch.tensor(nib.load(os.path.join(dir_name, mode, 'mr_labels', self.mri_label_names [i])).get_fdata().squeeze()) for i in range(self.num_data)]
        
    def __len__(self):
        return self.num_data
        
    def __getitem__(self, idx):
        
        #upsample_us = self.resample(self.us_data[idx])
        #upsample_us_labels = self.resample(self.us_labels[idx], label = True)
        
        if self.alligned:
            # no need to transpose if alligned alerady
            t_us = self.us_data[idx]
            t_us_labels = self.us_labels[idx]
            t_fake_us = self.fake_data[idx]
        else: # Change view of us image to match mr first 
            t_us = torch.transpose(self.us_data[idx], 2,0)
            t_us_labels = torch.transpose(self.us_labels[idx], 2,0)
            t_fake_us = torch.transpose(self.fake_data[idx], 2,0)
            
        # Upsample to MR images 
        upsample_us = self.resample(t_us)
        upsample_us_labels = self.resample(t_us_labels, label = True)
        upsample_fake = self.resample(t_fake_us)
        
        # Squeeze dimensions ; add batch dimension 
        us = upsample_us.squeeze().unsqueeze(0)
        us_label = upsample_us_labels.squeeze().unsqueeze(0)
        fake_us = upsample_fake.squeeze().unsqueeze(0)
        
        # normalise data 
        #mr = self.normalise_data(mr_data)
        #mr_label = self.normalise_data(mr_label)
        us = self.normalise_data(us)
        us_label = self.normalise_data(us_label)
        fake_us = self.normalise_data(fake_us)
        
        # if resample is true 
        if self.downsample: 
            upsample_method = torch.nn.Upsample(size = (64,64,64))
            #mr = upsample_method(mr.unsqueeze(0)).squeeze(0)
            #mr_label = upsample_method(mr_label.unsqueeze(0)).squeeze(0)
            us = upsample_method(us.unsqueeze(0)).squeeze(0)
            us_label = upsample_method(us_label.unsqueeze(0)).squeeze(0)
            fake_us = upsample_method(fake_us.unsqueeze(0)).squeeze(0)
        
        # Sample a slice and corresponding label 
        us_shape = us.size()
        
        # Sample fair; returns slice 50% with prostate, 50% without prostate 
        if self.sample_fair:
            WITH_PROSTATE = [idx for idx in range(us_shape[-1]) if torch.any(us_label[:,:,idx] > 0)]
            WITHOUT_PROSTATE = [idx for idx in range(us_shape[-1]) if not(torch.any(us_label[:,:,idx] > 0))]
            # Find index of slices WITH prostate 
            p = np.random.rand()
            # Find index of slices WITHOUT prostate
            if p < 0.5:
                # if p < 0.5 : with prostate
                slice_idx = np.random.choice(WITH_PROSTATE)
            else:
                # if p > 0.5 : without prostate
                slice_idx = np.random.choice(WITHOUT_PROSTATE)
            
            #print(f"Slice idx {slice_idx}")
            us_slice = us[:,:,slice_idx]
            fake_us_slice = fake_us[:,:,slice_idx]
            us_label_slice = us_label[:,:,slice_idx] # segmented slice 
            us_label_classify = torch.tensor(1.0*(len(torch.unique(us_label_slice)) > 1)) # 1 if presence of prostate 

            return us_slice, fake_us_slice, us_label_classify
            
        else:
            # Return entire volume 
            return us, fake_us, us_label
    
    def resample(self, img, dims = (120,128,128), label = False):
        upsample_method = torch.nn.Upsample(size = dims)
        if label: 
            
            if len(img.size()) > 4:
                img_label = img[:,:,:,:,0]
            else:
                img_label = img 
            
            # Note : upsample method requires dimensions BATCH_SIZE X NUM_CHANNELS X H X W X D (1x1x120x128x128)
            upsampled_img = upsample_method(img_label.unsqueeze(0))
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
    
    #data_folder = '/Users/ianijirahmae/Documents/DATASETS/mri_us_paired'
    #data_folder = './Data'
    #data_folder = './evaluate/REGNET'
    
    data_folder = './Data'
    train_dataset = US_dataset(data_folder, mode = 'train', downsample = True, alligned = False ) #, get_2d = True)
    train_dataloader = DataLoader(train_dataset)
    
    for idx, (mr, us, mr_label, us_label) in enumerate(train_dataloader):
        print('chicken')
        
    
    train_dataset = US_dataset(data_folder, mode = 'train')
    train_dataloader = DataLoader(train_dataset)
    
    us, us_label = train_dataset[0]
    
    for idx, (mr, us, mr_label, us_label) in enumerate(train_dataset):
        print(idx)
        fig, axs = plt.subplots(2,2)
        axs[0,0].imshow(mr[:,:,40])
        axs[0,1].imshow(mr_label[:,:,40,0])
        axs[1,0].imshow(us[:,:,80])
        axs[1,1].imshow(us_label[:,:,80])
        print('chicken')
        

        
    print('chicken')