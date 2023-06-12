import torch
import torch.nn as nn
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F

from dcgan import weights_init, Generator, Discriminator

# Set the path to the directory containing the JPEG images
image_dir = "monet_jpg/"

# Define image transformations (resize, normalize, etc.)
image_transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values
])

# Initialize an empty list to store the processed images
processed_images = []

ngpu = 1

# Loop through each JPEG image file in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Load the image from file
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)

        # Apply transformations to preprocess the image
        processed_image = image_transforms(image)

        # Append the processed image to the list
        processed_images.append(processed_image)

# Convert the list of processed images into a PyTorch tensor
input_data = torch.stack(processed_images)

# Print the shape of the input data
print("Input Data Shape:", input_data.shape)

device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("Using device", device)


########################################
## Create Generator and Discriminator ##
########################################

nc = 3  # no. of channels (rgb)
nz = 128 # size of latent (noise) vector
ngf = 64 # no. of filters/kernels for generator
ndf = 64 # no. of filters/kernels for discriminator

# Create the Generator
netG = Generator(ngpu, nc, nz, ngf).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Randomly initialize all weights
netG.apply(weights_init)

for param in netG.parameters(): # DELETE
    param.requires_grad = True

# Create the Discriminator
netD = Discriminator(ngpu, nc, ndf).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Randomly initialize all weights
netD.apply(weights_init)

for param in netG.parameters(): # DELETE
    param.requires_grad = True


#######################
## Create Optimizers ##
#######################

lr = 0.0002
beta1 = 0.5

# Initialize the BCELoss function
criterion = nn.BCELoss()

# Create batch of latent (noise) vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(nz, nz, 1, 1, device=device) # DELETE

# Establish convention for real and fake labels during training
real_label = 0.8
fake_label = 0.1

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


##############
## Training ##
##############

num_epochs = 300
batch_size = 16
lambda_reg = 0.0001

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(num_epochs):
    print('epoch: ', epoch)
    
    netG.train()
    netD.train()
    
    for i in range(int(np.ceil(input_data.shape[0]/batch_size))):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # Train Discriminator on separate mini-batches for real and fake samples
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = input_data[i*batch_size:(i+1)*batch_size].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label) + lambda_reg * sum(p.pow(2).sum() for p in netD.parameters())
        # Calculate gradients for D in backward pass
        errD_real.backward()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label) + lambda_reg * sum(p.pow(2).sum() for p in netD.parameters())
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake

        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label) + lambda_reg * sum(p.pow(2).sum() for p in netG.parameters())
        # Calculate gradients for G
        errG.backward()
        
        # Update G
        optimizerG.step()

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

####################
## Generate Image ##
####################

# call to create a Monet-style image
def generate_img(save=True):
    noise = torch.randn(1, 128, 1, 1, device=device)
    output = netG(noise).detach().cpu()
    output = (output + 1) / 2.0

    # Resize the image tensor to 256x256 using bilinear interpolation
    resized_img = F.interpolate(output, size=256, mode='bilinear', align_corners=False)

    res = transforms.ToPILImage()(resized_img.squeeze())

    if save:
        res.save("img.jpg")

    plt.imshow(res)
    plt.axis('off')
    plt.show()

generate_img()