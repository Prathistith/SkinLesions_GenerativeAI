import torch
from torch import nn
import torchvision
import torchvision.transforms as tt
import numpy as np
import matplotlib.pyplot as plt
from torchvision import *
import os
import time
from torchvision.utils import save_image
from PIL import Image
# Import discriminator and generator
from .model.discriminator import Discriminator
from .model.generator import Generator

#Import Datastet
from .dataset import Dataset
# import Utilities
from .utils import plot, w_distance, compute_gradient_penalty, penalty

# Basic set of Parameters 
image_size = (3, 128, 128)
grayscale = False
DATA_FOLDER = '03' # Folder where source data is loaded
plot_path='plots' # plots the LOSS
device = 'cuda' if torch.cuda.is_available() else 'cpu' # check For GPU
check_dir='model' # folder to store best model weights

# mean & standard deviation for Normalization and Denormalization.
mean=np.array([0.5,0.5,0.5])
std=np.array([0.5,0.5,0.5])
mean_t=torch.FloatTensor(mean).view(3,1,1).to(device)
std_t=torch.FloatTensor(std).view(3,1,1).to(device)


n_noise_features= 100 # Noise vector dimension
epochs= 10000 #Total Number of epochs for model training
disc_steps= 5
gen_steps= 1
lambda_pen= 10
discriminator_filters= 128
generator_filters= 128
batch_size= 32 # Image Batch size
print_every= 1
checkpoints= 500 # After this epoch model weights will be stored for future evaluation
rolling_window= 100
discriminator_label_noise= False
discriminator_input_noise= False
resume_training= None

# Dataset with transform & dataloader
transform=tt.Compose([tt.Resize((128,128)),tt.ToTensor(),tt.Normalize(mean,std)])
dset_train=Dataset(DATA_FOLDER,transform)
train_loader=torch.utils.data.DataLoader(dset_train,batch_size=32,shuffle=True)

# Discriminator & generator initialization
discriminator = Discriminator(image_size[0], discriminator_filters).to(device)
generator = Generator(n_noise_features, image_size[0], generator_filters).to(device)



############################################
###! loading model after certain epochs#####
############################################
# discriminator.load_state_dict(torch.load(check_dir+'/discriminator_<epoch>.pt'))
# generator.load_state_dict(torch.load(check_dir+'/generator<epoch>.pt'))
############################################

# utility function to save model state.
def checkpoint(disc, gen, epoch):
    if not os.path.isdir(check_dir):
        os.makedirs(check_dir)
    disc_dict = discriminator.state_dict()
    torch.save(disc_dict, f'model/discriminator_{epoch}.pt')
    gen_dict = generator.state_dict()
    torch.save(gen_dict, f'model/generator{epoch}.pt')

# Utility to generate random sample after certain epoch and check improvement on frame_noise.
def generate_sample(disc, gen, epoch,frame_noise):
    random_noise = torch.from_numpy(np.random.randn(batch_size, n_noise_features)).type(
            dtype=torch.FloatTensor).to(device)
    gen_output_random = generator(random_noise)*std_t+mean_t
    gen_output_frame = generator(frame_noise)*std_t+mean_t
    grid1 = utils.make_grid(gen_output_random.data.cpu()[:4])
    utils.save_image(grid1, 'training/img_generator_epoch_{}.png'.format(str(epoch)))
    grid12 = utils.make_grid(gen_output_frame.data.cpu())
    utils.save_image(grid12, 'image.png')

# Optimizers
disc_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=0.0001, betas=(0, 0.9))
gen_optimizer = torch.optim.Adam(
    generator.parameters(), lr=0.0001, betas=(0, 0.9))

# Lists to store losses
disc_losses, gen_losses, w_distances, gradient_penalty_list = [], [], [], []
gen_iterations = 0
steps = 0

# A noise vector where we generally visually check performance.
frame_noise = torch.from_numpy(np.random.randn(batch_size, n_noise_features)).type(
    dtype=torch.FloatTensor).to(device)

# Training
gen_min_loss = 99999.99
for e in range(epochs):
    if e % print_every == 0:
        print('Epoch {}'.format(e))
    start = time.time()
    epoch_dlosses, epoch_glosses = [], []
    train_iterator = iter(train_loader)
    i = 0
    while i < len(train_loader):
        noise_factor = (epochs - e) / epochs
        #########################
        # Train the discriminator
        #########################
        for p in discriminator.parameters():  # reset requires_grad
            p.requires_grad = True
        # train the discriminator disc_steps times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            disc_steps = 100
        else:
            disc_steps = disc_steps
        j = 0
        lll=0.0
        while j < disc_steps and i < len(train_loader):
            j += 1
            i += 1
            images= train_iterator.next()
            images = images.to(device)
            common_batch_size = min(batch_size, images.shape[0])
            disc_optimizer.zero_grad()
            noises = torch.from_numpy(np.random.randn(common_batch_size, n_noise_features)).type(
                dtype=torch.FloatTensor).to(device)
            # Compute output of both the discriminator and generator
            disc_output = discriminator(images)
            gen_images = generator(noises)
            gen_output = discriminator(gen_images)

            gradient_penalty = compute_gradient_penalty(images, gen_images, discriminator, lambda_pen)
            loss = torch.mean(gen_output - disc_output + gradient_penalty)
            loss.backward()
            wdist = torch.mean(disc_output - gen_output)
            disc_optimizer.step()

            # Save the loss
            lll+=loss.item()
            epoch_dlosses.append(loss.item())
            w_distances.append(wdist.item())
            gradient_penalty_list.append(torch.mean(gradient_penalty).item())
          
            steps += 1
        disc_losses.append(lll/j)

        #######################
        # Train the generator
        #######################
        for p in discriminator.parameters():  # reset requires_grad
            p.requires_grad = False
        gen_optimizer.zero_grad()
        noises = torch.from_numpy(np.random.randn(batch_size, n_noise_features)).type(
            dtype=torch.FloatTensor).to(device)
        gen_images = generator(noises)
        gen_output = discriminator(gen_images)

        loss = - torch.mean(gen_output)
       
        loss.backward()
        gen_optimizer.step()
        # Save the loss
        gen_losses.append(loss.item())
        #check if Loss if is less than min loss. If yes, take checkpoint.
        if(gen_min_loss>=loss.item() and loss.item() >=0.0):
            gen_min_loss=loss.item()
            checkpoint(discriminator,generator,-1)

        epoch_glosses.append(loss.item())
        gen_iterations += 1
    # take random samples after 100 epoch
    if e % 100 == 0:
        generate_sample(discriminator, generator, e, frame_noise)
        print('D loss: {:.5f}\tG loss: {:.5f}\tTime: {:.0f}'.format(
            np.mean(epoch_dlosses), np.mean(epoch_glosses), time.time() - start))

    # Take checkpoints after num of checkpoints epoch.
    if e % checkpoints == 0:
        checkpoint(discriminator, generator, e)
    # start plotting the loss
    plot(disc_losses,gen_losses)
    penalty(gradient_penalty_list)
    w_distance(w_distances)
