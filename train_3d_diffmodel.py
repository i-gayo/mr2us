from models.diffusion_utils_3d import * 
from utils.data_utils import * 
from models.diffusion_utils_2d import * 
from torch.utils.tensorboard import SummaryWriter
import os 
import argparse 
from torch.optim import Adam 
from PIL import Image
import h5py
from matplotlib import pyplot as plt 
from utils.augmentation_utils import transform
from utils.train_utils import save_3d_images

parser = argparse.ArgumentParser(prog='train',
                                description="Script for training classifier")

parser.add_argument('--log_dir',
                    '--log',
                    metavar='log_dir',
                    type=str,
                    action='store',
                    default='DIFF_AUGMENT',
                    help='Log dir to save results to')

parser.add_argument('--timesteps',
                    '--ts',
                    metavar='timesteps',
                    type=str,
                    action='store',
                    default='100',
                    help='Number of timesteps to use for denoising')

parser.add_argument('--batch_size',
                    '--bs',
                    metavar='batch_size',
                    type=str,
                    action='store',
                    default='2',
                    help='Batch size for sampling')

parser.add_argument('--save_h5',
                    '--h5',
                    metavar='save_h5',
                    type=str,
                    action='store',
                    default='False',
                    help='Whether to save h5 file for evaluating')

parser.add_argument('--scheduler',
                    metavar='scheduler',
                    type=str,
                    action='store',
                    default='linear',
                    help='Which scheduling type to use')

parser.add_argument('--objective',
                    metavar='objective',
                    type=str,
                    default='pred_x0',
                    action = 'store',
                    help='Objective function to use')

# partial, metric, none 
parser.add_argument('--loss_type',
                    metavar='loss_type',
                    type=str,
                    default='l2',
                    action = 'store',
                    help='Whether to include intensity in loss function : partial or none')

# Parse arguments
args = parser.parse_args()

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"]="2"
    DATA_FOLD = './evaluate/REGNET'
    BATCH_SIZE = int(args.batch_size)
    TIMESTEPS = int(args.timesteps)
    NUM_EPOCHS = 10000
    EVAL_FREQ = 10 # sample every 10 epochs 
    SAVE_H5 = (args.save_h5 == 'True')
    OBJECTIVE = args.objective 
    LOSS_TYPE = args.loss_type
    print(f"Batch size : {BATCH_SIZE} objective : {OBJECTIVE} Timesteps : {TIMESTEPS} LOSS : {LOSS_TYPE}")
    
    # Alligned = True means images are rotated correctly in the right way! 
    train_dataset = MR_US_dataset(DATA_FOLD, mode = 'train', downsample = True, alligned = True)
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    val_dataset = MR_US_dataset(DATA_FOLD, mode = 'val', downsample = True, alligned = True)
    val_dataloader = DataLoader(val_dataset, batch_size = 1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Initialising folders for saving models and images 
    BASE_FOLDER = 'DIFFUSION_MODELS' # where to save all models
    results_folder = os.path.join(BASE_FOLDER, args.log_dir)
    os.makedirs(results_folder, exist_ok = True)
    runs_folder = os.path.join(results_folder, 'runs')
    model_folder = os.path.join(results_folder, 'models')
    img_folder = os.path.join(results_folder, 'imgs')
    os.makedirs(runs_folder, exist_ok = True)
    os.makedirs(model_folder, exist_ok = True)
    os.makedirs(img_folder, exist_ok = True)
    
    #save_and_sample_every = 20 # every halfway through point, save model for each epoch. 
    
    # Initialising Unet and diffusion model
    model = create_model(
    image_size=64, 
    num_channels=32,
    num_res_blocks=2,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    in_channels=2,
    out_channels=1)
    model = model.to(device)
    diffusion_model = Diffusion_3D(model, device, scheduler_type = args.scheduler, timesteps = TIMESTEPS, objective = OBJECTIVE, loss_type = LOSS_TYPE)

    # Training process 
    optimiser = Adam(diffusion_model.parameters(), lr=1e-05) # changed from 1e-05'
    writer = SummaryWriter(runs_folder) 
    total_steps = 0 
    best_loss = 1000000
    best_loss_val = 1000000
    
    for epoch in range(NUM_EPOCHS):
        
        epoch_loss = [] 
        epoch_loss_val = []
        
        # metrics 
        epoch_ssim = []
        epoch_psnr = []
        epoch_nrmsd = [] 
        
        # Train 
        diffusion_model.train()              
        for step, (mr, us, mr_label, us_label) in enumerate(train_dataloader):

            #mr, us = mr.to(device), us.to(device)
            batch_size = mr.size()[0]
            
            # Transform mr2us 
            mrus = torch.cat((mr,us))
            mrus_transform = transform(mrus.squeeze())
            
            split_num = round(mrus_transform.size()[0]/2)
            mr_transform = mrus_transform[0:split_num,:,:,:].unsqueeze(1)#.unsqueeze(0)
            us_transform = mrus_transform[split_num:,:,:,:].unsqueeze(1)#.unsqueeze(0)
            
            mr_transform, us_transform = mr_transform.to(device), us_transform.to(device)
            
            # # Debugging 
            # save_folders = 'TRANSFORM_IMGS'
            # os.makedirs(save_folders, exist_ok = True)
            # mr_folder = os.path.join(save_folders, 'mr')
            # us_folder = os.path.join(save_folders, 'us')
            # os.makedirs(mr_folder, exist_ok = True)
            # os.makedirs(us_folder, exist_ok = True)
            # save_3d_images(mr_transform, save_folder = mr_folder, gt_imgs = mr)
            # save_3d_images(us_transform, save_folder = us_folder, gt_imgs = us)
            # save_3d_images(us_transform, save_folder = us_folder, gt_imgs = us)
                
            # Update gradients according to loss
            optimiser.zero_grad()
            t = torch.randint(0, TIMESTEPS, (batch_size,), device=device).long()
            loss, metrics = diffusion_model(us_transform.float(), mr_transform.float(), t)
            loss.backward()
            epoch_loss.append(loss.item())
            optimiser.step()
            
            # Save metrics 
            epoch_ssim.append(metrics['ssim'])
            epoch_psnr.append(metrics['psnr'])
            epoch_nrmsd.append(metrics['nrmsd'])
            
            print(f"step : {step} Loss : {loss.item()}")

            # Every 5th step, save images within training ; check eveyr 5 epochs 
            if (step % 10 == 0) and (epoch % 5 == 0): 
                generated_imgs = diffusion_model.sample(us.size(), mr_transform)
                print(f"Len of generated iamges {len(generated_imgs)} shape {np.shape(generated_imgs[0])}")
                save_name = str(epoch) + '_' + str(step)
                save_folder = os.path.join(img_folder, save_name)            
                save_3d_images(generated_imgs, save_folder, save_h5 = SAVE_H5)
        
        # Save most recent train model pth 
        train_model_path = os.path.join(model_folder, 'train_model.pth')
        torch.save(model.state_dict(), train_model_path)
        
        # Compute mean of metrics, add to writer!!! 
        e_ssim = np.nanmean(epoch_ssim)
        e_psnr = np.nanmean(epoch_psnr)
        e_nrmsd = np.nanmean(epoch_nrmsd)
        
        writer.add_scalar('Metrics/ssim', e_ssim, epoch)
        writer.add_scalar('Metrics/psnr', e_psnr, epoch)
        writer.add_scalar('Metrics/nrmsd', e_nrmsd, epoch)
             
        if step != 0 and (epoch % EVAL_FREQ == 0):
            
            # Evaluate model 
            diffusion_model.eval()     
            
            with torch.no_grad():
                
                for step, (mr, us, mr_label, us_label) in enumerate(val_dataloader):
                    
                    mr, us = mr.to(device), us.to(device)
                    t = torch.randint(0, TIMESTEPS, (mr.size()[0],), device=device).long()
                    loss_val, metrics_val = diffusion_model(us.float(), mr.float(), t)
                    epoch_loss_val.append(loss_val.item())
                    
                # Compute average loss for epoch 
                epoch_val = np.nanmean(epoch_loss_val)
                epoch_std_val = np.nanstd(epoch_loss_val)
                print(f"Epoch {epoch} mean val loss : {epoch_val} ± {epoch_std_val}")
                writer.add_scalar('Loss/val', epoch_val, epoch)

                if epoch_val < best_loss_val:
                    # Save and sample every 5 or 10 epochs  ; save best train model 
                    model_path = os.path.join(model_folder, ('best_train.pth'))
                    print(f"Saving model in model_path {model_path} for epoch {epoch} step {step}")
                    torch.save(model.state_dict(), model_path)
                    best_loss_val = epoch_val 
                   
                else:
                    # Save and sample every 5 or 10 epochs anyway  
                    model_path = os.path.join(model_folder, (str(epoch)+'_model.pth'))
                    print(f"Saving checkpoint {model_path} for epoch {epoch} step {step}")
                    torch.save(model.state_dict(), model_path)
                    
                    
                    
        total_steps+=1     
        
        with torch.no_grad():
                
            # Compute average loss for epoch 
            epoch_mean_loss = np.nanmean(epoch_loss)
            epoch_std_loss = np.nanstd(epoch_loss)
            print(f"Epoch {epoch} mean loss : {epoch_mean_loss} ± {epoch_std_loss}")
            writer.add_scalar('Loss', epoch_mean_loss, epoch)
            
            if epoch_mean_loss < best_loss:
                # Save new model 
                print(f"Saving new model with loss : {epoch_mean_loss}")
                model_path = os.path.join(model_folder, 'best_val_model.pth')
                torch.save(model.state_dict(), model_path)
                best_loss = epoch_mean_loss
                
        # Save most recent train model 
                  
        print('fuecoco')