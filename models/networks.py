import numpy as np 
import os 
import torch.nn as nn 
import torch 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import models.layers as layers # import model layers from Qianye's code 
import math 


######### simple transformnet based on unet architecture ########
class TransformNet(nn.Module):
    """
    A network that performs image-to-image translation / transformation 
    Network architecture is based on a 3D UNet 
    """
    
    def __init__(self, input_channel = 1, output_channel = 1, num_features = 32):
        super(TransformNet, self).__init__()

        self.num_features = num_features 

        # Identify each layers in the UNet
        self.encoder_1 = TransformNet._build_conv_block(input_channel, num_features)
        self.maxpool_1 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.encoder_2 = TransformNet._build_conv_block(num_features, num_features*2)
        self.maxpool_2 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.encoder_3 = TransformNet._build_conv_block(num_features*2, num_features*4)
        self.maxpool_3 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.encoder_4 = TransformNet._build_conv_block(num_features*4, num_features*8)
        self.maxpool_4 = nn.MaxPool3d(kernel_size = 2, stride = 2)

        self.bottle_neck = TransformNet._build_conv_block(num_features *8, num_features * 16)

        self.upconv_4 = nn.ConvTranspose3d(num_features * 16, num_features * 8, kernel_size = 2, stride = 2)
        self.decoder_4 = TransformNet._build_conv_block((num_features*8)*2, num_features *8)
        self.upconv_3 = nn.ConvTranspose3d(num_features * 8, num_features * 4, kernel_size = 2, stride = 2)
        self.decoder_3 = TransformNet._build_conv_block((num_features*4)*2, num_features *4)
        self.upconv_2 = nn.ConvTranspose3d(num_features * 4, num_features * 2, kernel_size = 2, stride = 2)
        self.decoder_2 = TransformNet._build_conv_block((num_features*2)*2, num_features *2)
        self.upconv_1 = nn.ConvTranspose3d(num_features*2 , num_features, kernel_size = 2, stride = 2)
        self.decoder_1 = TransformNet._build_conv_block(num_features*2, num_features)

        self.final_conv = nn.Conv3d(num_features, output_channel, kernel_size = 1)

    def forward(self, x):
        enc1 = self.encoder_1(x)
        enc2 = self.encoder_2(self.maxpool_1(enc1))
        enc3 = self.encoder_3(self.maxpool_2(enc2))
        enc4 = self.encoder_4(self.maxpool_3(enc3))

        bottleneck = self.bottle_neck(self.maxpool_4(enc4))

        dec4 = self.upconv_4(bottleneck)
        if dec4.size() == enc4.size():
            dec4 = torch.cat((dec4, enc4), dim=1)
        else: 
            #enc4_crop = self._crop(dec4, enc4)
            dec4 = self._pad(dec4, enc4)
            dec4 = torch.cat((dec4,enc4), dim = 1)
        dec4 = self.decoder_4(dec4)

        dec3 = self.upconv_3(dec4)
        if dec3.size() == enc3.size():
            dec3 = torch.cat((dec3, enc3), dim=1)
        else: 
            dec3 = self._pad(dec3, enc3)
            dec3 = torch.cat((dec3,enc3), dim = 1)
            
        dec3 = self.decoder_3(dec3)

        dec2 = self.upconv_2(dec3)
        if dec2.size() == enc2.size():
            dec2 = torch.cat((dec2, enc2), dim=1)
        else: 
            dec2 = self._pad(dec2, enc2)
            dec2 = torch.cat((dec2,enc2), dim = 1)
            
        dec2 = self.decoder_2(dec2)

        dec1 = self.upconv_1(dec2)
        if dec1.size() == enc1.size():
            dec1 = torch.cat((dec1, enc1), dim=1)
        else: 
            dec1 = self._pad(dec1, enc1)
            dec1 = torch.cat((dec1,enc1), dim = 1)
            
        dec1 = self.decoder_1(dec1)

        return torch.sigmoid(self.final_conv(dec1))

    @staticmethod
    def _build_conv_block(input_channel, num_features):
        
        conv_block = nn.Sequential(
            nn.Conv3d(in_channels = input_channel, out_channels = num_features, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm3d(num_features = num_features),
            nn.ReLU(inplace= True),
            nn.Conv3d(in_channels = num_features, out_channels = num_features, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm3d(num_features = num_features),
            nn.ReLU(inplace=True))

        return conv_block 
    
    def _crop(self, decoder, encoder):
        """
        A function that crops the encoder to be same size as decoder 
        """
        
        decode_size = decoder.size()
        h = decode_size[2] // 2 
        c = decode_size[3] // 2
        w = decode_size[4] // 2
        
        encode_size = encoder.size()
        h_e = encode_size[2] // 2
        c_e = encode_size[3] // 2
        w_e = encode_size[4] // 2
        
        crop_encoder = encoder[:,:,h_e - h: h_e+h, c_e - c:c_e+c, w_e - w:w_e+w]
        
        return crop_encoder 
    
    def _pad(self, decoder, encoder):
        """
        Pads decoder to be same size as encoder b
        """
        
        # calculate padding required
        padding_size = torch.tensor(encoder.size()) - torch.tensor(decoder.size())
        pad_list = tuple([x for item in padding_size for x in (item, 0)][:])
        # Pad size: 0,0,0,0,0,0 only for 
        padded_decoder = F.pad(decoder, (pad_list), 'constant', 0)
        
        return padded_decoder 

######## networks for pix2pix / cGAN ########
class Generator(nn.Module):
    """
    A network that performs image-to-image translation / transformation 
    Network architecture is based on a 3D UNet 
    """
    
    def __init__(self, input_channel = 1, output_channel = 1, num_features = 32, activation_layer = 'tanh'):
        super(Generator, self).__init__()

        self.num_features = num_features 
        self.activation_layer = activation_layer

        # Identify each layers in the UNet
        self.encoder_1 = TransformNet._build_conv_block(input_channel, num_features)
        self.maxpool_1 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.encoder_2 = TransformNet._build_conv_block(num_features, num_features*2)
        self.maxpool_2 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.encoder_3 = TransformNet._build_conv_block(num_features*2, num_features*4)
        self.maxpool_3 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.encoder_4 = TransformNet._build_conv_block(num_features*4, num_features*8)
        self.maxpool_4 = nn.MaxPool3d(kernel_size = 2, stride = 2)

        self.bottle_neck = TransformNet._build_conv_block(num_features *8, num_features * 16)

        self.upconv_4 = nn.ConvTranspose3d(num_features * 16, num_features * 8, kernel_size = 2, stride = 2)
        self.decoder_4 = TransformNet._build_conv_block((num_features*8)*2, num_features *8)
        self.upconv_3 = nn.ConvTranspose3d(num_features * 8, num_features * 4, kernel_size = 2, stride = 2)
        self.decoder_3 = TransformNet._build_conv_block((num_features*4)*2, num_features *4)
        self.upconv_2 = nn.ConvTranspose3d(num_features * 4, num_features * 2, kernel_size = 2, stride = 2)
        self.decoder_2 = TransformNet._build_conv_block((num_features*2)*2, num_features *2)
        self.upconv_1 = nn.ConvTranspose3d(num_features*2 , num_features, kernel_size = 2, stride = 2)
        self.decoder_1 = TransformNet._build_conv_block(num_features*2, num_features)

        self.final_conv = nn.Conv3d(num_features, output_channel, kernel_size = 1)

    def forward(self, x):
        enc1 = self.encoder_1(x)
        enc2 = self.encoder_2(self.maxpool_1(enc1))
        enc3 = self.encoder_3(self.maxpool_2(enc2))
        enc4 = self.encoder_4(self.maxpool_3(enc3))

        bottleneck = self.bottle_neck(self.maxpool_4(enc4))

        dec4 = self.upconv_4(bottleneck)
        if dec4.size() == enc4.size():
            dec4 = torch.cat((dec4, enc4), dim=1)
        else: 
            #enc4_crop = self._crop(dec4, enc4)
            dec4 = self._pad(dec4, enc4)
            dec4 = torch.cat((dec4,enc4), dim = 1)
        dec4 = self.decoder_4(dec4)

        dec3 = self.upconv_3(dec4)
        if dec3.size() == enc3.size():
            dec3 = torch.cat((dec3, enc3), dim=1)
        else: 
            dec3 = self._pad(dec3, enc3)
            dec3 = torch.cat((dec3,enc3), dim = 1)
            
        dec3 = self.decoder_3(dec3)

        dec2 = self.upconv_2(dec3)
        if dec2.size() == enc2.size():
            dec2 = torch.cat((dec2, enc2), dim=1)
        else: 
            dec2 = self._pad(dec2, enc2)
            dec2 = torch.cat((dec2,enc2), dim = 1)
            
        dec2 = self.decoder_2(dec2)

        dec1 = self.upconv_1(dec2)
        if dec1.size() == enc1.size():
            dec1 = torch.cat((dec1, enc1), dim=1)
        else: 
            dec1 = self._pad(dec1, enc1)
            dec1 = torch.cat((dec1,enc1), dim = 1)
            
        dec1 = self.decoder_1(dec1)

        if self.activation_layer == 'tanh':
            final_layer = torch.tanh(self.final_conv(dec1))
        elif self.activation_layer == 'relu':
            final_layer = torch.relu(self.final_conv(dec1))
        else: 
            # by default, use sigmoid 
            final_layer = torch.sigmoid(self.final_conv(dec1))
        return final_layer
    
    @staticmethod
    def _build_conv_block(input_channel, num_features):
        
        conv_block = nn.Sequential(
            nn.Conv3d(in_channels = input_channel, out_channels = num_features, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm3d(num_features = num_features),
            nn.ReLU(inplace= True),
            nn.Conv3d(in_channels = num_features, out_channels = num_features, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm3d(num_features = num_features),
            nn.ReLU(inplace=True))

        return conv_block 
    
    def _crop(self, decoder, encoder):
        """
        A function that crops the encoder to be same size as decoder 
        """
        
        decode_size = decoder.size()
        h = decode_size[2] // 2 
        c = decode_size[3] // 2
        w = decode_size[4] // 2
        
        encode_size = encoder.size()
        h_e = encode_size[2] // 2
        c_e = encode_size[3] // 2
        w_e = encode_size[4] // 2
        
        crop_encoder = encoder[:,:,h_e - h: h_e+h, c_e - c:c_e+c, w_e - w:w_e+w]
        
        return crop_encoder 
    
    def _pad(self, decoder, encoder):
        """
        Pads decoder to be same size as encoder b
        """
        
        # calculate padding required
        padding_size = torch.tensor(encoder.size()) - torch.tensor(decoder.size())
        pad_list = tuple([x for item in padding_size for x in (item, 0)][:])
        # Pad size: 0,0,0,0,0,0 only for 
        padded_decoder = F.pad(decoder, (pad_list), 'constant', 0)
        
        return padded_decoder 
   
class Discriminator(nn.Module):
    """
    Define a patch-gan level discriminator that outputs 70 x 70 
    
    Note: code is modified from original pix2pix implementaion found in :
    https://learnopencv.com/paired-image-to-image-translation-pix2pix/#discriminator
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    
    Args:
    --------------
    input_channel (int): Number of input channels.
    num_features (int, optional): Number of filters in the first convolutional layer (default is 64).
    n_layers (int, optional): Number of convolutional layers in the discriminator (default is 3).
    norm_layer (torch.nn.Module, optional): Normalization layer to be used (default is nn.BatchNorm2d).
    """
    def __init__(self, input_channel, num_features=64, n_layers=3, norm_layer=nn.BatchNorm3d):
        super(Discriminator, self).__init__()
        kernel_size = 4
        pad_width = 1
        
        sequence = [nn.Conv3d(input_channel, num_features, kernel_size=kernel_size, stride=2, padding = pad_width), nn.LeakyReLU(0.2, True)]
        
        # 
        filter_mult = 1
        filter_mult_prev = 1 # previous multiplier 
        
        for n in range(1, n_layers):  # gradually increase the number of filters
            filter_mult_prev = filter_mult
            filter_mult = min(2 ** n, 8) # increase up to 8, but not more. 
            sequence += [
                nn.Conv3d(num_features * filter_mult_prev, num_features * filter_mult, kernel_size=kernel_size, stride=2, padding=pad_width, bias=False),
                norm_layer(num_features * filter_mult),
                nn.LeakyReLU(0.2, True)
            ]

        filter_mult_prev = filter_mult
        filter_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(num_features * filter_mult_prev, num_features * filter_mult, kernel_size=kernel_size, stride=1, padding=pad_width, bias=False),
            norm_layer(num_features * filter_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Add final conv3d layer with sigmoid activation function 
        sequence += [nn.Conv3d(num_features * filter_mult, 1, kernel_size=kernel_size, stride=1, padding = pad_width), nn.Sigmoid()]  # output 1 channel prediction map
        
        # Model goes through entire sequence of layers 
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

####### networks for registration ########

class LocalNet(nn.Module):
    """
    Code adapted from mpmrieg.src.model.networks localnet, but removed config details for easier implementation
    """
    def __init__(self,input_shape, device, input_channel = 2, ddf_levels = [0, 1, 2, 3, 4], num_features = 32):
        super(LocalNet, self).__init__()
        self.input_shape = input_shape
        self.ddf_levels = ddf_levels
        self.device = device

        nc = [num_features*(2**i) for i in range(5)]
        up_sz = self.calc_upsample_layer_output_size(self.input_shape, 4)
        # print('up_sz', up_sz)

        self.downsample_block0 = layers.DownsampleBlock(inc=input_channel, outc=nc[0])
        self.downsample_block1 = layers.DownsampleBlock(inc=nc[0], outc=nc[1])
        self.downsample_block2 = layers.DownsampleBlock(inc=nc[1], outc=nc[2])
        self.downsample_block3 = layers.DownsampleBlock(inc=nc[2], outc=nc[3])
        self.conv_block = layers.Conv3dBlock(inc=nc[3], outc=nc[4])

        self.upsample_block0 = layers.UpSampleBlock(inc=nc[4], outc=nc[3], output_size=up_sz[0])
        self.upsample_block1 = layers.UpSampleBlock(inc=nc[3], outc=nc[2], output_size=up_sz[1])
        self.upsample_block2 = layers.UpSampleBlock(inc=nc[2], outc=nc[1], output_size=up_sz[2])
        self.upsample_block3 = layers.UpSampleBlock(inc=nc[1], outc=nc[0], output_size=up_sz[3])
        self.ddf_fuse_layers = [layers.DDFFusion(inc=nc[4 - i], out_shape=self.input_shape).to(self.device) for i in self.ddf_levels]


    def calc_upsample_layer_output_size(self, input_shape, num_downsample_layers=4):
        shape = np.array(input_shape)
        tmp = [list(shape//(2**i)) for i in range(num_downsample_layers)]
        tmp.reverse()

        # # Add channel depth and filter size to first two dimensions 
        # for i in range(len(tmp)):
        #     tmp[i][0] = 2
        #     tmp[i][1] = round(256/(2**i))
        
        return tmp 
    
    def forward(self, x):
        f_down0, f_jc0 = self.downsample_block0(x)
        f_down1, f_jc1 = self.downsample_block1(f_down0)
        f_down2, f_jc2 = self.downsample_block2(f_down1)
        f_down3, f_jc3 = self.downsample_block3(f_down2)
        f_bottleneck = self.conv_block(f_down3)

        # print(f_down0.shape, f_jc0.shape)
        # print(f_down1.shape, f_jc1.shape)
        # print(f_down2.shape, f_jc2.shape)
        # print(f_down3.shape, f_jc3.shape)
        # print(f_bottleneck.shape)

        f_up0 = self.upsample_block0([f_jc3, f_bottleneck])
        f_up1 = self.upsample_block1([f_jc2, f_up0])
        f_up2 = self.upsample_block2([f_jc1, f_up1])
        f_up3 = self.upsample_block3([f_jc0, f_up2])

        # print('-'*20)
        # print(f_up0.shape)
        # print(f_up1.shape)
        # print(f_up2.shape)
        # print(f_up3.shape)

        ddf0 = self.ddf_fuse_layers[0](f_bottleneck)
        ddf1 = self.ddf_fuse_layers[1](f_up0)
        ddf2 = self.ddf_fuse_layers[2](f_up1)
        ddf3 = self.ddf_fuse_layers[3](f_up2)
        ddf4 = self.ddf_fuse_layers[4](f_up3)

        ddf = torch.sum(torch.stack([ddf0, ddf1, ddf2, ddf3, ddf4], axis=5), axis=5)
        
        return f_bottleneck, ddf

#### Diffusion models 
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import nn, einsum


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
        self.alpha, self.beta, self.alpha_bar = self.compute_alphabetas(timesteps)
        
    
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
        
        return alphas, betas, alpha_bar
        
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

## DiffUNet building blocks 

def exists(x):
    return x is not None

class Residual(nn.Module):
    """"
    Computes residual connections between input and another layer 
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class CnnBlock(nn.Module):
    """
    A CNN block with time embeddings!!! 
    """
    def __init__(self, input_ch, output_ch, groups=8):
        super().__init__()
        self.conv = nn.Conv3d(in_channels = input_ch, out_channels = output_ch, kernel_size = 3, padding = 1, bias = False)
        self.norm = nn.GroupNorm(output_ch, groups)
        self.act = nn.SiLU()
                
    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x
    
class ResidualBlock(nn.Module):
    """
    Computes residual blocks for each UNet layer
    For each block : conv + groupnorm + silu activation layer 
    Time embed added before second convolution 
    """
    def __init__(self, input_ch, output_ch, time_emb_dim=None, groups = 8):
        """
        Residual layers 
        """
        super().__init__()
        self.input_ch = input_ch 
        self.output_ch = output_ch 
        self.num_groups = groups 
        
        # Time embeddings 
        self.time_emb_layers = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim,output_ch))
            if exists(time_emb_dim)
            else None)
        
        # Residual convolution
        self.residual_conv = nn.Conv3d(
            input_ch, out_channels=output_ch,
            kernel_size=1) if input_ch != output_ch else nn.Identity()

        self.conv_block1 = CnnBlock(input_ch, output_ch, groups = groups)
        self.conv_block2 = CnnBlock(output_ch, output_ch, groups = groups) 
    
    def forward(self, x, time_emb = None):
        """
        Forward pass for each block
        """
        
        res_out = self.residual_conv(x)
        
        scale_shift = None
        if exists(self.time_emb_layers) and exists(time_emb):
            time_emb = self.time_emb_layers(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        out = self.conv_block1(x, scale_shift)
        out = self.conv_block2(out)
        
        return out + res_out


        
        
    def _create_block(self, block_type = 'first'):
        
        input_num = self.input_ch
        if block_type == 'first':
            input_num = self.input_ch
        else:
            input_num = self.output_ch 
            
        block = nn.Sequential(
        nn.Conv3d(in_channels = input_num, out_channels = self.output_ch, kernel_size = 3, padding = 1, bias = False), 
        nn.GroupNorm(num_channels = self.output_ch, num_groups = self.num_groups),
        nn.SiLU())
        
        return block 
          
class Attention3D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) d h w -> b h c d h w", h=self.heads),
            qkv
        )
        q = q * self.scale

        # q * k 
        sim = einsum("b h d i j, b h d i k -> b h i j k", q, k)
        sim = sim - sim.amax(dim=(-1, -2, -3), keepdim=True).detach()
        
        # softmax(q*k)
        attn = F.softmax(sim, dim=(-1, -2, -3))
        
        # softmax(q*k) * v
        out = einsum("b h i j k, b h d k -> b h i d j", attn, v)
        out = rearrange(out, "b h c d h w -> b (h c) d h w")
        return self.to_out(out)
    
class LinearAttention3D(nn.Module):
    """
    Computes multi-head linear attention for 3D volumes. 
    
    Uses equation: 
    
    z = sofrtmax(Q x K^T / sqrt(dk)) * V 
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv3d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, d, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        
        # Maps qkv vectors to separate qkv 
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) d x y -> b h c d (x y)", h=self.heads), qkv
        )

        # Obtain soft max values for q and k separately 
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        
        # Obtain context = k*v 
        context = torch.einsum("b h d e n, b h d e m -> b h d n m", k, v)

        # obtains k*v*q vectors 
        out = torch.einsum("b h d n m, b h d e n -> b h e n m", context, q)
        
        # Re-arrange to number of heads
        out = rearrange(out, "b h c d (x y) -> b (h c) d x y", h=self.heads, x=d, y=h)
        
        return self.to_out(out)

class PreNorm(nn.Module):
    """
    Applies group norm before attention layerr 
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
   
## Diff UNet functions 

class DiffUNet(nn.Module):
    """
    A conditional UNEt, conditioned on both time steps 
    and MR images for diffusion generation!!!

    
    Note : 
    -----------
    Differences from normal unet include:
    - Attention module 
    - Residual blocks 
    - ResNet blocks 
    - Conditional concatenation from timestep / noise level t
    
    Input: Noisy image (us), cond_img (mr) and noise levels 
    - noisy images (concat) : batch_size x (num_channels*2), height, width, depth)
    - noise_levels : (batch_size, 1)
    
    output: Noise 
    - batch_size x num_channels, height, width, depth 
    """
    
    def __init__(self, 
                 init_dim, 
                 output_dim, 
                 feature_size, 
                 dim_mults = (1,2,4,8), 
                 channels = 2,
                 self_condition = False,
                 resnet_block_groups = 4):
        """
        Define internal layers
        
        1. Convolutional layer applied on noisy images; position embeddings computed
        2. Down-samplign stages 
            - (2 ResNet blocks + group norm + attent                                                                                                            ion + residual connection + downsample operation)
        3. ResNet blocks applied, interleaved with attention
        4. Upsampling stages:
            - 2 ResNet blocks; group norm ; attention ; residual connection ; upsapmle operation
        5. ResNet, followed by convolution
        """
        super().__init__()
        
        dim = 1 
        self.channels = channels 
        self.self_condition = self_condition 
        input_channels = channels * (2 if self_condition else 1)
        self.init_ch = init_dim
        self.output_ch = output_dim 

        # Compute hidden filter sizes
        self.layer_size = [feature_size * (2*i) for i in dim_mults]
    
        # Compute conv layer for noisy images (conditioned)
        self.first_conv = nn.Conv3d(input_channels, init_dim,1, padding = 0)
                
        # Compute position embeddings
        timestep_input_dim = self.layer_size[0] # same size as first layer size 
        embed_dim = timestep_input_dim * 4
        
        self.time_embed_layers = nn.Sequential(
            SinusoidalPositionEmbeddings(self.layer_size[0]),
            nn.Linear(timestep_input_dim, embed_dim), 
            nn.SiLU(), 
            nn.Linear(embed_dim, embed_dim))            
    
        # Downsample layers
        self.ds_layers = nn.ModuleList()
        self.us_layers = nn.ModuleList()
    
        in_dim = self.init_ch
        for idx, layer_size in enumerate(self.layer_size[0:]):
            self.ds_layers.append(
                nn.ModuleList(
                    [ResidualBlock(in_dim, layer_size, embed_dim),
                    ResidualBlock(layer_size, layer_size, embed_dim), 
                    Residual(PreNorm(layer_size, LinearAttention3D(layer_size))),
                    self.downsample_3d(layer_size, layer_size)] # downsample to next layer 
                )
            )
            in_dim = layer_size # change to next layer size 
        
        # Bottleneck ResNet blocks applied, interleaved with attention
        mid_layer_size = self.layer_size[-1]
        self.bottleneck_1 = ResidualBlock(mid_layer_size, mid_layer_size, embed_dim)
        self.bottleneck_attn = Residual(PreNorm(mid_layer_size, LinearAttention3D(mid_layer_size)))
        self.bottleneck_2 = ResidualBlock(mid_layer_size, mid_layer_size, embed_dim)
        
        # Upsample blocks 2 ResNet blocks; group norm ; attention ; residual connection ; upsapmle operation
        in_dim = mid_layer_size 
        for idx, layer_size in enumerate(list(reversed(self.layer_size[:-2]))):
            self.us_layers.append(
                nn.ModuleList(
                    [ResidualBlock(in_dim,  layer_size, embed_dim), 
                    ResidualBlock(in_dim,  layer_size, embed_dim), 
                    Residual(PreNorm(in_dim,LinearAttention3D(in_dim))),
                    self.upsample_3d(in_dim, layer_size)    ]
                )
            )
            in_dim = layer_size 
            
        self.final_res = ResidualBlock(self.layer_size[0]*2, self.layer_size[0], embed_dim)
        self.final_conv = nn.Conv3d(self.layer_size[0], self.output_ch, kernel_size = 1)
        
    def forward(self, x, time):
        if not torch.is_tensor(time):
            timesteps = torch.tensor([time],
                                     dtype=torch.long,
                                     device=x.device)
        if self.self_condition:
            x_self_cond = torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.first_conv(x)
        r = x.clone() # for residual connections 

        t = self.time_embed_layers(time)

        h = []

        for block1, block2, attn, downsample in self.ds_layers:
            x = block1(x, t)
            h.append(x) # Append h 

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.bottleneck_1(x, t)
        x = self.bottleneck_attn(x)
        x = self.bottleneck_2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res(x, t)
        return self.final_conv(x)


        
    def downsample_3d(self, input_ch, output_ch):
        """
        Downsamples layers 
        """        
        
        return nn.Sequential(
            # Adjusting the spatial and channel rearrangement for 3D
            Rearrange("b c d (h p1) (w p2) -> b (c p1 p2) d h w", p1=2, p2=2),
            nn.Conv3d(input_ch * 4, output_ch, kernel_size=1),
        )

    def upsample_3d(self, input_ch, output_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(input_ch, output_ch, 3, padding=1),
        )
        
    def sinusoidal_embedding(self, timesteps, dim):
        """
        Create sinusidal embeddings for each time step 
        
        Tells us at which timestep the noise is injected 
        
        PE(pos, 2i) = sin(pos / 1000^2i / d_model)
        PE(pos, 2i+1) = cos(pos / 1000^2i/d_model)

        where pos is position of token (timestep)
        i is dimension of embedding 
        d_model : dimensionality of model or embedding        
        
        scaling facotr : 10000
        
        """
        
        # half_dim = dim // 2
        
        # # Based on expression : -log(10000) / (half_dim -1)
        # exponent = -math.log(10000) * torch.arange(
        #     start=0, end=half_dim, dtype=torch.float32)
        # exponent = exponent / (half_dim - 1.0)

        # embed = torch.exp(exponent).to(device=timesteps.device)
        # embed = timesteps[:, None].float() * embed[None, :]
        
        # Re-wriitng in my own words -> pos / 1000^2i / d_model
        half_dim = dim // 2
        SCALE_FACTOR = 10000
        embeddings = math.log(SCALE_FACTOR) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings) 
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # sine and cosine EMBEDDINGS 
        

        return embeddings
    



if __name__ == '__main__':
    
    #from ..utils.data_utils import * 
    
    BATCH_SIZE = 2
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    regnet = LocalNet([120,120,20], device = device)
    # Load dataloaders, models 
    data_folder = './Data'
    train_dataset = MR_US_dataset(data_folder, mode = 'train')
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    val_dataset = MR_US_dataset(data_folder, mode = 'val')
    val_dataloader = DataLoader(val_dataset, batch_size = 1)
    
    # Testing fissuion process 
    mr, us, mr_label, us_label = train_dataset[0]
    
    # Test diffusion
    DiffProcess = Diffusion('linear', timesteps = 300)
    DiffProcess.get_noisy_img(mr, t = torch.tensor([40]))
    
    # Define models for training 
    regnet = LocalNet([120,120,20], device = device)
    discriminator_net = Discriminator(input_channel = 2) # channels being cat layers 
    generator_net = Generator(input_channel = 1) # input is MR image only; output we want is US image 
    use_cuda = True
    save_folder = 'pix2pix'
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    trainer = Pix2pixTrainer(generator_net, discriminator_net, train_dataloader, val_dataloader, device = device, log_dir = 'pix2pix_test')
    trainer.train()
        
    
    for idx, (mr, us, mr_label, us_label) in enumerate(train_dataset):
        
        # stack along channel dimension!!!
        combined_mr_us = torch.cat((mr.unsqueeze(0), us.unsqueeze(0).unsqueeze(0)), 1) # cat on channel dimesnion 
        
        # Outputs 13 x 14 x 14 patches 
        discrim_labels = discriminator_net(combined_mr_us.float())
        generated_img = generator_net(mr.unsqueeze(0).float())
        
        print('chicken')

    