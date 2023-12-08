import torch 
import numpy as np 
from utils.data_utils import MR_US_dataset
from torchmetrics.functional.image import structural_similarity_index_measure as ssim 
from torch.utils.tensorboard import SummaryWriter
import os 
from torch.utils.data import DataLoader, Dataset
from models.networks import TransformNet, LocalNet, Generator
from matplotlib import pyplot as plt 
import nibabel
import torch
import nibabel as nib
import numpy as np
import mpmrireg.src.model.functions as smfunctions

def make_data_folders(specific_folder):
    """
    A function that makes train, val folders with images
    """
    
    for mode in ['train', 'val']:
        # make train and test folder 
        mode_folder = os.path.join(specific_folder, mode)
        os.makedirs(mode_folder, exist_ok = True)
        
        for word in ['mr_images', 'mr_labels', 'us_images', 'us_labels']:
            new_folder = os.path.join(mode_folder, word)
            os.makedirs(new_folder, exist_ok = True)

        
def save_nifti(tensor, filename, affine=None):
    """
    Save a PyTorch tensor as a NIfTI image.

    Parameters:
        - tensor: PyTorch tensor to be saved as a NIfTI image.
        - filename: Output file path for the NIfTI image.
        - affine: (Optional) Affine transformation matrix.
                  If None, an identity matrix will be used.

    Note:
        - The tensor should have dimensions (C, D, H, W), where
          C is the number of channels, D is the depth, H is the height,
          and W is the width.
    """
    # Convert PyTorch tensor to NumPy array
    data_array = tensor.cpu().numpy()

    # Check and correct the shape if necessary
    if data_array.ndim == 3:
        # Add a singleton dimension for the channel
        data_array = np.expand_dims(data_array, axis=0)
    elif data_array.ndim != 4:
        raise ValueError("Invalid tensor shape. Expected 3D or 4D tensor.")

    # Create a NIfTI image object
    nifti_img = nib.Nifti1Image(data_array, affine)

    # Save the NIfTI image to file
    nib.save(nifti_img, filename)
    
def get_warped_images(move_label, ddf):

    warped_us = smfunctions.warp3d(move_label, ddf)

    return warped_us 

if __name__ == '__main__':

    MODEL = 'regnet'
    BATCH_SIZE = 1
    SAVE_FOLDER = 'evaluate'
    MODE = 'train'
    use_cuda = False 
    # Load dataloaders, models 
    data_folder = './Data'
    train_dataset = MR_US_dataset(data_folder, mode = 'train')
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    val_dataset = MR_US_dataset(data_folder, mode = 'val')
    val_dataloader = DataLoader(val_dataset, batch_size = 1)
    
    if MODE == 'train':
        data = train_dataset
    else:
        data = val_dataset
    
    # Getting dataset size 
    mr, us, mr_label, us_label = train_dataset[0]

    #new_us = torch.transpose(us, 3, 1)
    #mr = torch.flip(mr, [-1])
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt 
    
    # Plotting figures
    fig, axs = plt.subplots(1,2)
    
    # for idx in range(1, 128, 5):
    #     axs[0].imshow(mr.squeeze()[:,:,idx])
        
    #     axs[1].imshow(us.squeeze()[:,:,idx])
    #     plt.pause(0.5)
    
    mr_us = torch.cat([mr, us], dim=0).float()
    input_shape = (mr_us.size()[-3:])

    # Obtain device 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Device : {device}")
    
    if MODEL == 'transform':
        # Define model  
        model = TransformNet()
        use_cuda = False
        specific_folder = os.path.join(SAVE_FOLDER, 'transform')
        os.makedirs(specific_folder, exist_ok=True) 
        
        # make folders for train, val 
        #make_data_folders(specific_folder)
                
        MODEL_PATH = '/Users/ianijirahmae/Documents/PhD_project/Experiments/mr2us_exp/NEW_MODELS/transformnet_v2/best_val_model.pth'
    
    elif MODEL == 'regnet':
        model = LocalNet(input_shape, device)
        specific_folder = os.path.join(SAVE_FOLDER, 'regnet')
        os.makedirs(specific_folder, exist_ok=True) 

        # make data folders 
        make_data_folders(specific_folder)
        MODEL_PATH = '/Users/ianijirahmae/Documents/PhD_project/Experiments/mr2us_exp/NEW_MODELS/regnet_v3/checkpoints/Epoch-690.pt'
                        
    elif MODEL == 'pix2pix':
        model = Generator()
        specific_folder = os.path.join(SAVE_FOLDER, 'pix2pix')
        os.makedirs(specific_folder, exist_ok=True) 
        
        # make folders for saving 
        #make_data_folders(specific_folder)
        MODEL_PATH = '/Users/ianijirahmae/Documents/PhD_project/Experiments/mr2us_exp/NEW_MODELS/pix2pix_v3/gen_model.pth'
        
    # Load model 
    model.load_state_dict(torch.load(MODEL_PATH, map_location = 'cpu'))
    
    # Inference script 
    with torch.no_grad():
        
        # Evaluate 
        for idx, (mr, us, mr_label, us_label) in enumerate(data):
                    
            if use_cuda: 
                mr, us = mr.cuda(), us.cuda()
        
            if MODEL != 'regnet':
                preds = model(mr.float().unsqueeze(0))
            else:
                mr_us = torch.cat([mr, us], dim=0).float()
                
                # Obtain DDF by obtaining input data
                ddf = model(mr_us.unsqueeze(0))
                ddf_img = ddf[1]
                
                # Warp masks (not input images) 
                preds = get_warped_images(us.unsqueeze(0), ddf_img)
                pred_labels = get_warped_images(us_label.unsqueeze(0), ddf_img)
                
            # Write preds to nii file 
            img_num = str(idx)
            if len(img_num) == 1:
                case_name = 'case00000'
            else: 
                case_name = 'case0000'
                
            # Save us and mr images separately : preds is aligned images 
            img_name = case_name + img_num + '.nii.gz'
            
            if MODEL != 'regnet':
                file_path = os.path.join(specific_folder, img_name)
                save_nifti(preds.squeeze(), file_path)
            
            else: 
                # Save predicted image, labels 
                folder_path = os.path.join(specific_folder, MODE, 'us_images', img_name)
                save_nifti(preds.squeeze(), folder_path)
                
                folder_path = os.path.join(specific_folder, MODE, 'us_labels', img_name)
                save_nifti(pred_labels.squeeze(), folder_path)
                
                folder_path = os.path.join(specific_folder, MODE, 'mr_images', img_name)
                save_nifti(mr.squeeze(), folder_path)

                folder_path = os.path.join(specific_folder, MODE, 'mr_labels', img_name)
                save_nifti(mr_label.squeeze(), folder_path)
            
            print('chicken')


            num_slices = preds.size()[-1]
            
            # if MODEL != 'regnet':
            #     gt = us 
            #     fig, axs = plt.subplots(1,2)
            #     for i in range(0, num_slices, 5):
            #         axs[0].imshow(preds.squeeze()[:,:,i])
            #         axs[1].imshow(gt.squeeze()[:,:,i])
            #         plt.pause(0.5)
                
            # else:
                
            #     fig, axs = plt.subplots(1,3)
                
            #     for i in range(0, num_slices, 5):
            #         axs[0].imshow(preds.squeeze()[:,:,i])
            #         axs[1].imshow(mr.squeeze()[:,:,i])
            #         axs[2].imshow(us.squeeze()[:,:,i])
            #         plt.pause(0.5)
                    
                
                    

                
        print('chickne')      
    