import torch
from models.generator import Generator
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
G.load_state_dict(torch.load(r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Görüntü İşleme\Projeler\Renklendirme\saved_models2\generator_epoch238.pth", map_location=device))
G.eval()

img = Image.open(r"C:\Users\Casper\OneDrive\Masaüstü\flower.jpg").convert("L")

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    y = G(x).squeeze().cpu() * 0.5 + 0.5

out = transforms.ToPILImage()(y.clamp(0, 1))
out.save(r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Görüntü İşleme\Projeler\Renklendirme\results2\generated_238_f4.jpg")









