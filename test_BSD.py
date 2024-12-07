import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from BSD import BSDDataset

base_dir = "/home/jingpu/Projects/Wavelet_proj"
noise_level = 10
# model_name = "DAE_"+str(noise_level)
model_name = "DnCNN_"+str(noise_level)

test_set = BSDDataset(base_dir=base_dir, split="test")
test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)

output_dir = os.path.join(base_dir, "results_BSD", model_name)
original_path = os.path.join(output_dir, "original_images")
noisy_path = os.path.join(output_dir, "noisy_images")
denoised_path = os.path.join(output_dir, "denoised_images")
os.makedirs(original_path, exist_ok=True)
os.makedirs(noisy_path, exist_ok=True)
os.makedirs(denoised_path, exist_ok=True)
model = torch.load(os.path.join(output_dir, "model.pt"))
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model.to(device)

counter = 0
for j, data in enumerate(test_loader, 0):
    images, _ = data
    noisy_images = images + (noise_level/255)*torch.randn(*images.shape)
    noisy_images = np.clip(noisy_images, 0, 1)
    images = images.to(device) # move to GPU
    noisy_images = noisy_images.to(device)
    outputs = model(noisy_images) # forward
    images = images.cpu().detach().numpy()
    noisy_images = noisy_images.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    for i in range(len(images)):
        image = 255 * np.transpose(images[i], (1,2,0))
        noisy_image = 255 * np.transpose(noisy_images[i], (1,2,0))
        output = 255 * np.transpose(outputs[i], (1,2,0))
        cv2.imwrite(os.path.join(original_path, str(counter)+".png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(noisy_path, str(counter)+".png"), cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(denoised_path, str(counter)+".png"), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        counter += 1
    if counter > 100:
        break

PSNR_noisy = []
SSIM_noisy = []
PSNR_denoised = []
SSIM_denoised = []
for i in range(len(os.listdir(original_path))):
    image = cv2.imread(os.path.join(original_path, str(i)+".png"), 0)
    noisy_image = cv2.imread(os.path.join(noisy_path, str(i)+".png"), 0)
    denoised_image = cv2.imread(os.path.join(denoised_path, str(i)+".png"), 0)
    PSNR_noisy.append(peak_signal_noise_ratio(image, noisy_image))
    SSIM_noisy.append(structural_similarity(image, noisy_image))
    PSNR_denoised.append(peak_signal_noise_ratio(image, denoised_image))
    SSIM_denoised.append(structural_similarity(image, denoised_image))
print("PSNR noisy:", format(np.mean(PSNR_noisy), ".2f"), "+-", format(np.std(PSNR_noisy), ".2f"))
print("SSIM noisy:", format(np.mean(SSIM_noisy), ".3f"), "+-", format(np.std(SSIM_noisy), ".3f"))
print("PSNR denoised:", format(np.mean(PSNR_denoised), ".2f"), "+-", format(np.std(PSNR_denoised), ".2f"))
print("SSIM denoised:", format(np.mean(SSIM_denoised), ".3f"), "+-", format(np.std(SSIM_denoised), ".3f"))