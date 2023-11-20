import numpy as np 
import os 
import torch.nn as nn 
import torch 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

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
         
class Pix2pixTrainer():
    """
    Defines a pix2pix trainer, with loss functions for both generator, discriminator 
    """

    def __init__(self, generator, discriminator, train_ds, val_ds, device, log_dir = 'pix2pix', lr_d = 1e-05, lr_g = 1e-05, lamda = 100):
        """
        Initializes the Pix2PixTrainer.

        Parameters:
        - generator (torch.nn.Module): The Pix2Pix generator network.
        - discriminator (torch.nn.Module): The Pix2Pix discriminator network.
        - train_ds (torch.utils.data.DataLoader): DataLoader for the training dataset.
        - val_ds (torch.utils.data.DataLoader): DataLoader for the training dataset.
        - device (str, optional): Device to use ('cuda' or 'cpu'). Default is 'cuda'.
        - log_Dir (str, optional) : Folder to save results into for tensorboard writing 
        - lr_d (float, optional): Learning rate for the optimizer. Default is 1e-05
        - lr_g (float, optional): Learning rate for the optimizer. Default is 1e-05
        - lamda (float, optional): Weight for the L1 loss term. Default is 100.
        """
        # Define cuda 
        self.device = device 
        self.log_dir = log_dir 
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialise sumnary writer and model paths 
        self.writer = SummaryWriter(self.log_dir) 
        self.gen_model_path = os.path.join(self.log_dir, 'gen_model.pth')
        self.dis_model_path = os.path.join(self.log_dir, 'dis_model.pth')
                
        # Define generators / discriminators 
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        
        # Define datasets 
        self.train_ds = train_ds
        self.val_ds = val_ds
        
        # Define loss functions
        self.adversarial_loss = nn.BCELoss()
        self.l1_loss_fn = nn.L1Loss()   
        self.lamda = 100
        
        # Define optimisers 
        self.D_opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d)
        self.G_opt = torch.optim.Adam(self.generator.parameters(), lr=lr_g)

    def train(self, NUM_EPOCHS = 1000, save_freq = 10):
        """
        A function to train pix2pix modelss
        """
        
        # Define discriminator and generator optimisers, loss functions
        
        for epoch in range(1, NUM_EPOCHS):
            
            print(f"\n Epoch number : {epoch}")
            D_loss_all, G_loss_all = [], [] 
            
            # Load datasets
            for idx, (mr, us, mr_label, us_label) in enumerate(self.train_ds):
                
                mr, us = mr.to(self.device), us.to(self.device)
                mr = mr.float()
                us = us.float()
                ##### Train discriminator #####
                self.D_opt.zero_grad()
                generated_img = self.generator(mr)
                
                # Compute loss with fake image 
                fake_cond_img = torch.cat((mr, generated_img), 1)
                D_fake = self.discriminator(fake_cond_img.detach())
                fake_label = torch.zeros_like(D_fake).to(self.device)
                D_fake_loss = self.discriminator_loss(D_fake, fake_label)

                # Compute loss with real image 
                real_cond_img = torch.cat((mr, us), 1)
                D_real = self.discriminator(real_cond_img) 
                real_label = torch.ones_like(D_real).to(self.device)
                D_real_loss = self.discriminator_loss(D_real, real_label)
                
                # Average discriminator loss 
                D_combined_loss = (D_real_loss + D_fake_loss) / 2
                D_loss_all.append(D_combined_loss)
                D_combined_loss.backward()
                self.D_opt.step()
                
                ##### Train generator with real labels #####
                self.G_opt.zero_grad()
                fake_img = self.generator(mr)    # Obtain fake img 
                fake_gen_img = torch.cat((mr, fake_img), 1)
                pred_label = self.discriminator(fake_gen_img)
                
                G_loss = self.generator_loss(fake_img, us, pred_label, real_label)
                G_loss_all.append(G_loss)
                
                G_loss.backward()
                self.G_opt.step()
                
                # Print discriminator loss and generator loss 
                print(f"Epoch {epoch} : Discriminator loss {D_combined_loss} Generator loss {G_loss}")
            
            # Save model every save_freq episodes 
            if epoch % save_freq: 
                torch.save(self.generator.state_dict(), self.gen_model_path)
                torch.save(self.discriminator.state_dict(), self.dis_model_path)
                
            # Compute mean loss for d and g
            with torch.no_grad():
                mean_d_loss = torch.mean(torch.tensor(D_loss_all))
                mean_g_loss = torch.mean(torch.tensor(G_loss_all))
            
            # Print and log loss values 
            print(f"Epoch finish training : mean d loss : {mean_d_loss} mean g loss : {mean_g_loss}")
            self.log_metrics(epoch, mean_d_loss, mean_g_loss)
        
        # Close summary writer 
        self.writer.close()

    def log_metrics(self, epoch, d_loss, g_loss):
        """
        A function that logs loss values onto writer
        """
        self.writer.add_scalar('Discriminator_loss', d_loss, epoch)
        self.writer.add_scalar('Generator_loss', g_loss, epoch)
                        
    def generate_samples(self, input_images, num_samples = 5):
        """
        Generates samples using the trained Pix2Pix generator.

        Parameters:
        - input_images (torch.Tensor): Input images for generating samples.
        - num_samples (int, optional): Number of samples to generate. Default is 5.

        Returns:
        - torch.Tensor: Generated samples.
        """
        with torch.no_grad():
            self.generator.eval()
            input_images = input_images.to(self.device)
            generated_images = self.generator(input_images)
            self.generator.train()

        # Return a few generated samples
        return generated_images[:num_samples]
                          
    def generator_loss(self, gen_img, real_img, pred_label, real_label, lamda = 100):
        """
        Calculates the generator loss for a Pix2Pix GAN.

        Parameters:
        - gen_img (torch.Tensor): Generated image by the generator.
        - real_img (torch.Tensor): Real image from the target domain.
        - pred_label (torch.Tensor): Predictions of the discriminator for the generated image.
        - real_label (torch.Tensor): Ground truth labels for real images.
        - lamda (float, optional): Weight for the L1 loss term. Default is 100.

        Returns:
        - torch.Tensor: Total generator loss, combining adversarial and L1 losses.
        """
        
        #adversarial_loss = nn.BCELoss()
        #l1_loss_fn = nn.L1Loss()   
        
        gen_loss = self.adversarial_loss(pred_label, real_label)
        l1_loss = self.l1_loss_fn(gen_img, real_img)
        
        total_loss = gen_loss + lamda*l1_loss 
        
        return total_loss

    def discriminator_loss(self,output_image, label):
        """
        Calculates the discriminator loss for a Pix2Pix GAN.

        Parameters:
        - output_image (torch.Tensor): Output image produced by the generator.
        - label (torch.Tensor): Ground truth labels for the corresponding images.

        Returns:
        - torch.Tensor: Discriminator loss based on the adversarial loss between
        the generated output and the ground truth labels.
        """
        
        #adversarial_loss = nn.BCELoss()
        discrim_loss = self.adversarial_loss(output_image, label)
        
        return discrim_loss 
