
To train model on CIFAR-10/BSD-500:
Specify the directory, number of epoch, noisy level (currently we comapre 10, 25, 50), model name at the beginning of "train_CIFAR.py"/"train_BSD.py" and run it, the train/test loss and trained model will be saved.

To test the trained model on CIFAR-10/BSD-500:
Specify the directory, noisy level (currently we comapre 10, 25, 50), model name at the beginning of "test_CIFAR.py"/"test_BSD.py" and run it, it will print the SSIM/PSNR before and after denoising. The original, noisy and denoised images will be saved.

For CIFAR-10: You may need to set download=True in the tv.datasets.CIFAR10 when you first run it.
For BSD-500: Put BSD500 folder under data/

DAE, UNet and DnCNN are used as baseline models, to add a new model, just import it and train/test using the pipeline.
