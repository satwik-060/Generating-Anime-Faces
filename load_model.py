#import stuff
import torch 
import torch.nn  as nn
import torchvision 
import torchvision.transforms as transforms
import torch.nn.functional as F 
from torchvision.utils import save_image

import numpy as np

# used for normalization 
stats = (0.5,0.5,0.5) , (0.5,0.5,0.5)

# to denorm while we are inferencing 
def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

#to setup default device, if we have gpu then we can use it for faster calculations.

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

device = get_default_device()

#hyperparameters
nc = 3 #no  of channels in training images.
latent_size = 100 # size of latent vector
ngf = 64 # size of feature maps in generator
ndf = 64 # size of feature maps in discriminator
num_epochs = 30# no of epochs the training is done
lr = 0.0002 # learning rate for Adam
beta1 = 0.5 # beta1 hyperparameter of Adam Optimizer


#Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            
            nn.ConvTranspose2d( latent_size, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )
        self.main = to_device(self.main,device)
    def forward(self, input):
        return self.main(input)
    


netG = Generator().to(device)
netG.load_state_dict(torch.load('models/G.pth'))