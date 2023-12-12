import numpy as np 
import os 
import torch.nn as nn 
import torch 
import torch.nn.functional as F
from pathlib import Path
import tqdm

## Diffusion functions 
class BetaSchedulers():
    """
    A class of beta schedulers for use in diffusion process 
    """
    def __init__(self, scheduler_type = 'linear'):
        """
        Define which scheduler to use
        """
        self.scheduler_type = scheduler_type
    
    def get_schedule(self, timesteps):
        
        if self.scheduler_type == 'linear':
            return self.linear_beta_schedule(timesteps)
        elif self.scheduler_type == 'cosine':
            return self.cosine_beta_schedule(timesteps)
        elif self.scheduler_type == 'quadratic':
            return self.quadratic_beta_schedule(timesteps)
        elif self.scheduler_type == 'sigmoid':
            return self.sigmoid_beta_schedule(timesteps)
        else: 
            raise NotImplementedError("Scheduler type not recognised, please enter valid type")

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Generate a cosine learning rate schedule as proposed in the paper:
        "Positional Embeddings improve Transformer-based Monaural Speech Separation"
        (https://arxiv.org/abs/2102.09672)

        Args:
        - timesteps (int): Number of timesteps or steps in the schedule.
        - s (float, optional): Scaling factor for the cosine schedule. Defaults to 0.008.

        Returns:
        - torch.Tensor: Beta values representing the schedule.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def linear_beta_schedule(self, timesteps):
        """
        Generate a linear learning rate schedule.

        Args:
        - timesteps (int): Number of timesteps or steps in the schedule.

        Returns:
        - torch.Tensor: Beta values representing the schedule.
        """
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    def quadratic_beta_schedule(self, timesteps):
        """
        Generate a quadratic learning rate schedule.

        Args:
        - timesteps (int): Number of timesteps or steps in the schedule.

        Returns:
        - torch.Tensor: Beta values representing the schedule.
        """
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

    def sigmoid_beta_schedule(self, timesteps):
        """
        Generate a sigmoidal learning rate schedule.

        Args:
        - timesteps (int): Number of timesteps or steps in the schedule.

        Returns:
        - torch.Tensor: Beta values representing the schedule.
        """
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

class Diffusion(nn.Module):
    """
    Methods to apply diffusion noise to images for the forward diffusion process 
    """

    def __init__(self, scheduler_type = 'linear', timesteps = 300):
        """
        """

        self.scheduler = BetaSchedulers(scheduler_type)
        self.alpha, self.beta, self.alpha_bar, self.alpha_bar_prev, self.post_var = self.compute_alphabetas(timesteps)
        self.timesteps = timesteps 
    
    def compute_alphabetas(self, timesteps):
        """
        Compute alphas for diffusion modelling 
        
        Note : Alpa and beta are known 
        """
        
        betas = self.scheduler.get_schedule(timesteps)
        alphas = 1 - betas 
        
        
        alphas = alphas
        betas = betas 
        alpha_bar = torch.cumprod(alphas, axis = 0) # alpha bar which is used for q(x_t | x_o)
        
        # Compute variables for computation of mean / variance later 
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        post_var = betas * (1. - alpha_bar_prev) / (1. - alpha_bar)
        
        
        return alphas, betas, alpha_bar, alpha_bar_prev, post_var
        
    def q_sample(self, x_o, t, noise = None):
        """
        Function that applies forward diffusion process to sample a noisy image 
        at noise level t, given x_o 
        
        Equation : 
        q(x_t | x_o) = normal(x_t ; mean = sqrt(alpha_bar_t)x_o, variance = (a-alpha_bar_t)I)
        
        Returns x_t based on q(x_t) equation 
        """
        
        if noise is None: 
            # no noise given as input 
            noise = torch.randn_like(x_o)
            
            
        mean_val_t = self.extract_t(torch.sqrt(self.alpha_bar), t, x_o.size()) 
        var_t = self.extract_t(torch.sqrt(1 - self.alpha_bar), t, x_o.size())
        
        # q(x_t | x_o) = Normal distribution (x_t ; sqrt(alpha_bar_t)*x_o, (1 - alpha_bar_t)I)
        x_t = mean_val_t * x_o + var_t*noise 
        
        return x_t 
            
    def extract_t(self, a, t, x_shape):
        """
        Extract t values for given alpha 
        """
        batch_size = t.shape[0]
        
        # Obtain t values 
        out = a.gather(-1, t.cpu())
        
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def get_noisy_img(self, x_o, t):
        
        # obtain noise
        x_t = self.q_sample(x_o, t = t)
        
        return x_t 

    def p_sample(self, model, x, cond_mr, t, t_index):
        """
        Computes revesre process 
        """
        with torch.no_grad():
            beta_t = self.extract_t(self.beta, t, x.size())
            sqrt_recip_alphas = torch.sqrt(1.0 / self.alpha)
            
            sqrt_1min_alpha_t = self.extract_t(torch.sqrt(1-self.alpha_bar), t, x.size())
            sqrt_recip_alpha_t = self.extract_t(sqrt_recip_alphas, t, x.size())
            
            # Obtain model mean 
            device = cond_mr.get_device()
            x = x.to(device)
            concat_img = torch.cat((x,cond_mr), axis = 1).float().to(device)
            model_mean = sqrt_recip_alpha_t * (x - beta_t * model(concat_img, t) / sqrt_1min_alpha_t)
            
            if t_index == 0: 
                return model_mean 
            
            else: 
                # using equation 11 
                pos_var_t = self.extract_t(self.post_var, t, x.size())
                noise = torch.randn_like(x)
                x_t_1 = model_mean + torch.sqrt(pos_var_t)*noise 
        
                return x_t_1
    
    def p_sample_loop(self, model, shape, cond_mr):
        
        device = next(model.parameters()).device
        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, cond_mr, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs

    def sample(self, model, image_size, cond_mr):
        
        return self.p_sample_loop(model, image_size, cond_mr)
     
    def p_loss(self, model, x_start, cond_mr, t, noise=None, loss_type = 'l1'):
        """
        A loss function that computes loss between predicted and output noise 
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        #Obtain noisy image of x, given start x
        x_noisy = self.q_sample(x_start, t=t, noise=noise)
        concat_img = torch.cat((x_noisy,cond_mr), axis = 1).float().to(device)
        pred_noise = model(concat_img, t)
        
        if loss_type == 'l1':
            loss = torch.nn.functional.l1_loss(pred_noise, noise)
        elif loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
        elif loss_type == 'huber':
            loss = torch.nn.functional.smooth_l1_loss(pred_noise, noise)
        else:
            raise NotImplementedError()
        
        return loss  
    
class DiffTrainer():
    """
    Trainer for training diffusion process 
    """
    
    def p_loss(self, pred_noise, gt_noise, loss_type = 'l1'):
        """
        A loss function that computes loss between predicted and output noise 
        """
        
        if loss_type == 'l1':
            loss = torch.nn.functional.l1_loss(pred_noise, gt_noise)
        elif loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(pred_noise, gt_noise)
        elif loss_type == 'huber':
            loss = torch.nn.functional.smooth_l1_loss(pred_noise, gt_noise)
        else:
            raise NotImplementedError()
        
        return loss 
    
results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 1000

from torch.optim import Adam 
device = "cuda" if torch.cuda.is_available() else "cpu"
