# mr2us
A repository which will conduct image-to-image translation from MR to US slices; converting pre-operative MR slices to simulated intra-operative US slices 


## Dataset

Paired 3d volumes of MRI and US images are used as the data. However, these images are not registered. 

## Purpose
The purpose of this repository is to investigate different methods to translate MR images to US images, with or without registration 

## Methods

Different methods will be investigated: 
* RegNet : A registration network that aims to align paired MR-US images 
* TransformNet : A basic encoder-decoder model based on the UNet architecture, which aims to translate MR images (as input) to US images (as output). The input and output pairs will be paired, but can be un-aligned. 
* Pix2Pix : A conditional generative adversarial network type architecture, aiming to produce US images, by conditioning on MR images.
* Diffusion models : (TBC) uses diffusion models to perform image-to-image translation 
