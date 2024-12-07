import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BSDDataset(Dataset):
    def __init__(self, base_dir, split):
        self.image_dir = os.path.join(base_dir, "data", "BSD500", split)
        self.image_files = sorted(os.listdir(self.image_dir))
        self.transform = transforms.Compose([ 
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), []
