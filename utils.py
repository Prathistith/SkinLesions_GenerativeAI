'''
'compute_gradient_penalty': calculates gradient penalty for WGAN

'plot': Used for plotting generator loss, discriminator loss vs number of epoch

'penalty': plot gradient penalty

'w_distance': plot Wasserstein distance vs number of epoch
''

'''
#IMPORTS
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as utils

plot_path='plots' # plots the LOSS
device = 'cuda' if torch.cuda.is_available() else 'cpu' # check For GPU

def compute_gradient_penalty(real, fake, discriminator, lambda_pen):
    # Compute the sample as a linear combination
    alpha = torch.rand(real.shape[0], 1, 1, 1).to(device)
    alpha = alpha.expand_as(real)
    x_hat = alpha * real + (1 - alpha) * fake
    # Compute the output
    x_hat = torch.autograd.Variable(x_hat, requires_grad=True)
    out = discriminator(x_hat)
    # compute the gradient relative to the new sample
    gradients = torch.autograd.grad(
        outputs=out,
        inputs=x_hat,
        grad_outputs=torch.ones(out.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    # Reshape the gradients to take the norm
    gradients = gradients.view(gradients.shape[0], -1)
    # Compute the gradient penalty
    penalty = (gradients.norm(2, dim=1) - 1) ** 2
    penalty = penalty * lambda_pen
    return penalty


def plot(val_loss,train_loss):
    plt.title("Loss after epoch: {}".format(len(train_loss)))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(list(range(len(train_loss))),train_loss,color="r",label="Generator_loss",alpha=0.7)
    plt.plot(list(range(len(val_loss))),val_loss,color="b",label="Discriminator_loss",alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(plot_path,"loss_model.png"))
    plt.close()

def penalty(loss):
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(list(range(len(loss))),loss,color="r",label="Gradient_penalty",alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(plot_path,"penalty.png"))
    plt.close()

def w_distance(loss):
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(list(range(len(loss))),loss,color="r",label="w_distance",alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(plot_path,"w_distance.png"))
    plt.close()