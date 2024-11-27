import os
import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from DAE import DAE
from DnCNN import DnCNN

base_dir = "/home/jingpu/Projects/Wavelet_proj"
n_epoch = 30
noise_level = 50
model = DAE(num_channels=3)
model_name = "DAE_"+str(noise_level)
# model = DnCNN(num_channels=3, num_layers=17)
# model_name = "DnCNN_"+str(noise_level)

transform = transforms.Compose([ # original size: [3, 32, 32]
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# (50000, 2); each item: (img, label)
train_set = tv.datasets.CIFAR10(root=os.path.join(base_dir, "data"),
                                train=True,
                                download=False,
                                transform=transform)
# (10000, 2)
test_set = tv.datasets.CIFAR10(root=os.path.join(base_dir, "data"),
                               train=False,
                               download=False,
                               transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, 
                                           shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, 
                                          shuffle=True, num_workers=4)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model.to(device)

train_loss_all = []
test_loss_all = []

for epoch in tqdm(range(n_epoch)):
    # training
    train_loss = 0
    model.train()
    for data in train_loader:
        images, _ = data
        noisy_images = images + (noise_level/255)*torch.randn(*images.shape)
        noisy_images = np.clip(noisy_images, 0, 1)
        images = images.to(device) # move to GPU
        noisy_images = noisy_images.to(device)
        optimizer.zero_grad()
        outputs = model(noisy_images) # forward
        outputs = outputs.to(device)
        loss = criterion(outputs, images)
        loss.backward() # backward
        optimizer.step() # update parameters
        train_loss += loss.item()
    print("Epoch", epoch+1, "training loss:", format(train_loss, ".3f"))
    train_loss_all.append(train_loss/len(train_set))
    # testing
    test_loss = 0
    test_PSNR = 0
    test_SSIM = 0
    model.eval()
    for data in test_loader:
        images, _ = data
        noisy_images = images + (noise_level/255)*torch.randn(*images.shape)
        noisy_images = np.clip(noisy_images, 0, 1)
        images = images.to(device) # move to GPU
        noisy_images = noisy_images.to(device)
        outputs = model(noisy_images) # forward
        outputs = outputs.to(device)
        loss = criterion(outputs, images)
        test_loss += loss.item()
    print("Epoch", epoch+1, "testing loss:", format(test_loss, ".3f"))
    test_loss_all.append(test_loss/len(test_set))

# save model and results
output_dir = os.path.join(base_dir, "results", model_name)
os.makedirs(output_dir, exist_ok=True)
torch.save(model, os.path.join(output_dir, "model.pt"))
np.save(os.path.join(output_dir, "train_loss.npy"), np.array(train_loss_all))
np.save(os.path.join(output_dir, "test_loss.npy"), np.array(test_loss_all))

