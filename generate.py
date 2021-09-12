# IMPORTS
import torch
from torch import nn
import torchvision
import torchvision.transforms as tt
import numpy as np
import matplotlib.pyplot as plt
from torchvision import *
import os
from torchvision.utils import save_image
from PIL import Image

# loading Generator
from .model.generator import Generator

image_size = (3, 128, 128)


store_path='<where to store>' # where output images should be stored.
NUM_GEN=200 # Number of generated samples required
device = 'cuda' if torch.cuda.is_available() else 'cpu' # check For GPU

# For Denormalization
mean=np.array([0.5,0.5,0.5])
std=np.array([0.5,0.5,0.5])
mean_t=torch.FloatTensor(mean).view(3,1,1).to(device)
std_t=torch.FloatTensor(std).view(3,1,1).to(device)

generator=Generator(100, image_size[0], image_size[1]).to(device)
# initialize generator weights with best model.
generator.load_state_dict(torch.load('<load best model weigts>'))

random_noise = torch.from_numpy(np.random.randn(NUM_GEN, 100)).type(
            dtype=torch.FloatTensor).to(device)
gen_output_random = generator(random_noise)*std_t+mean_t

j=0
for i in gen_output_random:
    save_image(i,store_path+f'gen_img{j}.png')
    j+=1

