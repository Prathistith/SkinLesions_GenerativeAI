# IMPORTS
import os
import torch
from torchvision import transforms
from PIL import Image

'''
Custom Dataset Class to load images and perform conversion to tensor.
'''
class Dataset(torch.utils.data.Dataset):
    def __init__(self,image_path,transform=None):
        self.filename = os.listdir(image_path)
        self.image_path = image_path
        if(transform == None):
            self.transform=transforms.ToTensor()
        else:
            self.transform=transform


    def __getitem__(self,index):
        image=self.transform(Image.open(self.image_path+'/'+self.filename[index]))
        return(image) # C x H x W


    def __len__(self):
        return(len(self.filename))
