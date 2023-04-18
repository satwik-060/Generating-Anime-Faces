from load_model import netG,denorm,device,latent_size
import torch
from torchvision.utils import save_image

latent = torch.randn(1,latent_size,1,1,device = device)
fake_image = netG.forward(latent)
fake_fname = 'generated_image.png'
save_image(denorm(fake_image),fake_fname)