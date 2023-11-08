import numpy as np 
import os 
import torch.nn as nn 
import torch 
import torch.nn.functional as F

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
    
class RegNet():
    """
    TODO : implement registraiton network 
    """
    pass
