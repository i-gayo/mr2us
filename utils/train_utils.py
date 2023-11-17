import numpy as np 
import os 
import torch.nn as nn 
import torch 
import torch.nn.functional as F


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


    def train(self, NUM_EPOCHS = 1000):
        """
        A function to train pix2pix modelss
        """
        
        # Define discriminator and generator optimisers, loss functions
        
        for epoch in range(1, NUM_EPOCHS):
            
            D_loss_all, G_loss_all = [], [] 
            
            # Load datasets
            for idx, (mr, us, mr_label, us_label) in enumerate(self.train_ds):
                
                if self.use_cuda():
                    mr, us = mr.to(self.device), us.to(self.device)
                
                    ##### Train discriminator #####
                    self.D_opt.zero_grad()
                    generated_img = self.generator(mr)
                    
                    # Compute loss with fake image 
                    fake_cond_img = torch.cat((mr, generated_img), 1)
                    D_fake = self.discriminator(fake_cond_img.detach())
                    fake_label = torch.Variable(torch.zeros_like(D_fake).to(self.device))
                    D_fake_loss = self.discriminator_loss(D_fake, fake_label)

                    # Compute loss with real image 
                    real_cond_img = torch.cat((mr, us), 1)
                    D_real = self.discriminator(real_cond_img) 
                    real_label = torch.Variable(torch.ones_like(D_real).to(self.device))
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
