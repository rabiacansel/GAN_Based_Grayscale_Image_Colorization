import torch
from models.generator import Generator
from torchvision import transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = r;

output_dir = r;
os.makedirs(output_dir, exist_ok=True)

image_path = r;
img = Image.open(image_path).convert("L")

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
x = transform(img).unsqueeze(0).to(device)

for model_file in os.listdir(model_dir):
    if model_file.endswith(".pth") and "generator" in model_file.lower():
        model_path = os.path.join(model_dir, model_file)

        try:
            G = Generator().to(device)
            G.load_state_dict(torch.load(model_path, map_location=device))
            G.eval()

            with torch.no_grad():
                y = G(x).squeeze().cpu() * 0.5 + 0.5

            out = transforms.ToPILImage()(y.clamp(0, 1))
            model_name = os.path.splitext(model_file)[0]
            save_path = os.path.join(output_dir, f"{model_name}.jpg")
            out.save(save_path)
            print(f"{model_name} çıktısı kaydedildi.")
        
        except Exception as e:
            print(f"{model_file} modeli yüklenemedi: {e}")
