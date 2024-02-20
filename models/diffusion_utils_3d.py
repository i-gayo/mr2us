#### Code obtained from repository https://github.com/mobaidoctor/med-ddpm/tree/main
#-*- coding:utf-8 -*-
# *Main part of the code is adopted from the following repository: https://github.com/openai/guided-diffusion

import numpy as np 
import os 
import torch.nn as nn 
import torch 
import torch.nn.functional as F
from pathlib import Path
import tqdm
from functools import partial
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import math
from torch import nn, einsum
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from models.fp16_util import convert_module_to_f16, convert_module_to_f32
from models.diffusion_utils_2d import * 
from models.modules import *
from models.losses import * 

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


class Diffusion_3D(nn.Module):
    """"
    Implements DIFFUSION MODELS FOR 2d models 
    
    Based off code : https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/guided_diffusion.py#L380
    
    Differences:
    - Removed loss weight
    - Removed prediction of v (only predicts noise or x_start)
    
    """
    def __init__(self, 
                 model,
                 device,
                 timesteps = 100,
                 scheduler_type = 'linear',
                 objective = 'pred_noise', 
                 loss_type = 'l2'
                 ):
    
        # Parameters
        super().__init__()
        self.timesteps = timesteps 
        self.scheduler = BetaSchedulers(scheduler_type=scheduler_type)
        self.objective = objective 
        self.model = model 
        self.device = device
        self.loss_type = loss_type
        
        betas = self.scheduler.get_schedule(timesteps)
        alphas = 1 - betas 
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        
        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32).to(self.device))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))


        # for q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        
        # Start t = 0
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        # From after 
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    #### new functions I dont have
    def predict_start_from_noise(self, x_t, t, noise):
        """
        Get x_0 from noise epsilon.

        The reparameterization gives:
            x_t = sqrt(alphas_cumprod) * x_0
                + sqrt(1-alphas_cumprod) * epsilon
        so,
        x_0 = 1/sqrt(alphas_cumprod) * x_t
            - sqrt(1-alphas_cumprod)/sqrt(alphas_cumprod) * epsilon
            = 1/sqrt(alphas_cumprod) * x_t
            - sqrt(1/alphas_cumprod - 1) * epsilon
            
        x_0 = coeff_t * x*t - coeff_noise *noise
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        """Get noise epsilon from x_0 and x_t.

        The reparameterization gives:
            x_t = sqrt(alphas_cumprod) * x_0
                + sqrt(1-alphas_cumprod) * epsilon
        so,
        epsilon = (x_t - sqrt(alphas_cumprod) * x_0) / sqrt(1-alphas_cumprod)
                = (1/sqrt(alphas_cumprod) * x_t - x_0)
                  /sqrt(1/alphas_cumprod-1)
        """    
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Computes mean adn variance of q(x_t | x_t-1) 
        # Computes mean and variance of q(x_t, x_t-1) ie distirbution where data came from
        
        """
        
        #mean of q(x_{t-1} | x_t, x_0)
        # mean = coeff_t0 * x_start + coeff_t * x_t 
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        
        # Compute variance and log variance (clipped) 
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_sample(self, x_start, t, noise = None):
        """
        Samples x_t from q(x_t | x_0)
        q(x_t | x_o) = normal(x_t ; mean = sqrt(alpha_bar_t)x_o, variance = (a-alpha_bar_t)I)
        Returns noisy x_t based on q(x_t) equation 
        """
        
        noise = default(noise, lambda: torch.randn_like(x_start))
        q_mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        q_std = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) 
        
        x_t = q_mean * x_start + q_std *noise 
        
        return x_t
    
    def model_predictions(self, x, t, cond_img):
        """
        
        Computes model perdictions
        
        Args : 
            x : Noise-added image (x_o with noise applied at level t)
            t : noise level 
            cond_img : conditioned MR image to concatinate on noise added image 
            
        Returns: 
            pred_noise : predicted noise 
            x_start : input x_0 
        
        # Output can either be one of two things:
        # a: noise 
        # b : x_0 (original input image!)
        """
        concat_img = torch.cat((x, cond_img), axis = 1).float().to(self.device)        
        model_output = self.model(concat_img, t)

        if self.objective == 'pred_noise':
            # Predicts output as input noise applied (which will be used to denoise)
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        elif self.objective == 'pred_x0':
            # Predicts output as input image x0
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def p_mean_variance(self, x, t, cond_input, clip_denoised = True):
        """
        Computes mean of distribution p(x_(t-1)|x_t)
        
        """
        pred_noise, x_start = self.model_predictions(x, t, cond_input)

        # Compute model mean 
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    #### What I have from before: 
    @torch.no_grad()
    def p_sample(self, x, t: int, cond_img):
        """
        Computes backward process 
        """
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, variance, model_log_variance, x_start = self.p_mean_variance(
            x = x, t = batched_times, cond_input = cond_img)

        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        
        return pred_img, x_start
    
    @torch.no_grad()
    def p_sample_loop(self, shape, cond_mr, return_all_timesteps = False):
        #batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device = cond_mr.device)
        imgs = [img]

        x_start = None

        for t in reversed(range(0, self.timesteps)):#, desc = 'sampling loop time step', total = self.timesteps):
            img, x_start = self.p_sample(img, t, cond_mr)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        #ret = self.unnormalize(ret)
        
        return ret

    @torch.no_grad()
    def sample(self, image_size, cond_mr):
        
        return self.p_sample_loop(image_size, cond_mr)
     
    def p_losses(self, x_start, cond_mr, t, noise=None, loss_type = 'l2'):
        """
        A loss function that computes loss between predicted and output noise 
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Sample noise 
        x_noisy = self.q_sample(x_start, t=t, noise=noise)
        concat_img = torch.cat((x_noisy,cond_mr), axis = 1).float().to(self.device)
        output = self.model(concat_img, t)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
            
        if self.loss_type == 'l1':
            loss = torch.nn.functional.l1_loss(output, target)
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(output, target)
        elif self.loss_type == 'huber':
            loss = torch.nn.functional.smooth_l1_loss(output, target)
        elif self.loss_type == 'combined':
            # Compute combined intensity + MSE loss 
            coeff = 0.2
            mse_loss = torch.nn.functional.mse_loss(output, target)
            intensity_loss = torch.mean(psnr(output, target))
            loss = mse_loss - coeff*intensity_loss # want to maximise psnr_val, so use -
        elif self.loss_type == 'intensity':
            # Compute intensity loss only, without MSE 
            loss = - torch.mean(psnr(output, target)) # want to maximise psnr, so use - of loss function!!! 
        else:
            raise NotImplementedError()
        
        # Compute metrics
        with torch.no_grad():
        # Compute metrics for image intensities tracking 
        
            metrics = {}
            metrics['psnr'] = (torch.mean(psnr(output, target))).item() # compute mean across batch 
            
            # Higher, better 
            metrics['ssim'] = (torch.nanmean(ssim_3d(output, target))).item()
            
            # Lower; better
            metrics['nrmsd'] = (torch.mean(nrmsd(output, target))).item() # provided as just value not 
            
            # Higher PSNR, SSIM; LOWER NRMSD
            print(f"PSNR : {(metrics['psnr'])} SSIM : {(metrics['ssim'])} nrmsd : {(metrics['nrmsd'])}")
        
        # Return loss,mean
        #loss = loss.mean() # Compute mean across batch sizes 
        
        return loss, metrics
    
    def forward(self, img, t, cond_mr):
        """
        Computes fvorward process
        """
        b, c, h, w, d = img.size()
        #device = img.device
        #assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        #t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        #img = self.normalize(img)
        return self.p_losses(img, t, cond_mr)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=3,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
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
    in_channels=1,
    out_channels=1,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(1*out_channels if not learn_sigma else 2*out_channels),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )
    
if __name__ == '__main__':
    
    model = UNetModel(image_size = 128, in_channels = 2)
    
    print('fuecoco')