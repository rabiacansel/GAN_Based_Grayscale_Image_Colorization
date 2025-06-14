import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random

class GrayscaleColorDataset(Dataset):
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.gray_dir = os.path.join(base_dir, "dataset", "grayscale")
        self.color_dir = os.path.join(base_dir, "dataset", "rgb_image")

        self.gray_images = sorted([
            f for f in os.listdir(self.gray_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.color_images = sorted([
            f for f in os.listdir(self.color_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        assert len(self.gray_images) == len(self.color_images), \

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.gray_images)

    def __getitem__(self, idx):
        gray_path = os.path.join(self.gray_dir, self.gray_images[idx])
        color_path = os.path.join(self.color_dir, self.color_images[idx])

        gray = Image.open(gray_path).convert("L")
        color = Image.open(color_path).convert("RGB")

        gray = gray.resize((1024, 1024))
        color = color.resize((1024, 1024))

        if random.random() < 0.5:
            gray = gray.transpose(Image.FLIP_LEFT_RIGHT)
            color = color.transpose(Image.FLIP_LEFT_RIGHT)

        angle = random.uniform(-10, 10)
        gray = gray.rotate(angle)
        color = color.rotate(angle) 

        gray_tensor = self.to_tensor(gray)   # [1, H, W]
        color_tensor = self.to_tensor(color) # [3, H, W]

        return gray_tensor, color_tensor
