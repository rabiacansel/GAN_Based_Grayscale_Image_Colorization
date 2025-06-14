import torch
from models.generator import Generator
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
G.load_state_dict(torch.load(r", map_location=device))
G.eval()

img = Image.open(r).convert("L")

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    y = G(x).squeeze().cpu() * 0.5 + 0.5

out = transforms.ToPILImage()(y.clamp(0, 1))








