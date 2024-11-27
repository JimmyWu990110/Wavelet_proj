
To train model on CIFAR-10:
Specify the directory, number of epoch, noisy level (currently we comapre 10, 25, 50), model name at the beginning of "train_CIFAR.py" and run it, the train/test loss and trained model will be saved.

To test the trained model on CIFAR-10:
Specify the directory, noisy level (currently we comapre 10, 25, 50), model name at the beginning of "test_CIFAR.py" and run it, it will print the SSIM/PSNR before and after denoising. The original, noisy and denoised images will be saved.

You may need to set download=True in the tv.datasets.CIFAR10 when you first run it.

DAE, UNet and DnCNN are used as baseline models, to add a new model, just import it and train/test using the pipeline.

I will add BSD dataset later.
