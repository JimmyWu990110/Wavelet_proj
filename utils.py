import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity




def train(model, optimizer, epochs, train_set, test_set, batch_size, model_name, compute_loss, base_dir=""):
    train_loss_all = []
    test_loss_all = []
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    bar = tqdm(range(epochs))
    
    for epoch in bar:
        # training
        train_loss = 0
        model.train()
        for i, data in enumerate(train_loader):
            images, _ = data
            optimizer.zero_grad()
            loss = compute_loss(model, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            bar.set_postfix({"Step": str(epoch * len(train_loader) + i + 1) + "/" + str(epochs * len(train_loader)), "training loss": format(train_loss / (i+1), ".3f")})
        train_loss_all.append(train_loss/len(train_set))
        # testing
        test_loss = 0
        model.eval()
        for data in test_loader:
            images, _ = data
            with torch.no_grad():
                loss = model.loss(images)
            test_loss += loss.item()
        bar.set_postfix({"Epoch": epoch+1, "testing loss": format(test_loss, ".3f")})
        test_loss_all.append(test_loss/len(test_set))
    
    # save model and results
    output_dir = os.path.join(base_dir, "results_BSD", model_name)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model, os.path.join(output_dir, "model.pt"))
    np.save(os.path.join(output_dir, "train_loss.npy"), np.array(train_loss_all))
    np.save(os.path.join(output_dir, "test_loss.npy"), np.array(test_loss_all))
    
    plt.plot(train_loss_all, label="training loss")
    plt.plot(test_loss_all, label="testing loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss.png"), format="png")


def test(model, test_set, batch_size, model_name, noise_level, denoise, base_dir="", **kwargs):
    print("noise_level:", noise_level)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    output_dir = os.path.join(base_dir, "results_BSD", model_name + str(noise_level))
    original_path = os.path.join(output_dir, "original_images")
    noisy_path = os.path.join(output_dir, "noisy_images")
    denoised_path = os.path.join(output_dir, "denoised_images")
    os.makedirs(original_path, exist_ok=True)
    os.makedirs(noisy_path, exist_ok=True)
    os.makedirs(denoised_path, exist_ok=True)
    
    
    model.eval()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    counter = 0
    with torch.no_grad():
        for j, data in tqdm(enumerate(test_loader, 0)):
            images, _ = data
            noisy_images = images + (noise_level/255)*torch.randn(*images.shape)
            noisy_images = np.clip(noisy_images, 0, 1)
            images = images.to(device) # move to GPU
            noisy_images = noisy_images.to(device)
            outputs = denoise(model, noisy_images, **kwargs) # forward
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