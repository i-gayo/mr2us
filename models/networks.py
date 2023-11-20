import numpy as np 
import os 
import torch.nn as nn 
import torch 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

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

# networks for diffusion models ########

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

if __name__ == '__main__':
    
    from train import * 
    
    BATCH_SIZE = 2
    
    # Load dataloaders, models 
    data_folder = './Data'
    train_dataset = MR_US_dataset(data_folder, mode = 'train')
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    val_dataset = MR_US_dataset(data_folder, mode = 'val')
    val_dataloader = DataLoader(val_dataset, batch_size = 1)
    
    # Define models for training 
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

    