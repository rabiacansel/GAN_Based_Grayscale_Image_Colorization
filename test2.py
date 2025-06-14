import torch
from models.generator import Generator
from torchvision import transforms
from PIL import Image
import os

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model dosyalarının bulunduğu klasör
model_dir = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Görüntü İşleme\Projeler\Renklendirme\saved_models2"

# Çıktıların kaydedileceği klasör
output_dir = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Görüntü İşleme\Projeler\Renklendirme\all_model_results"
os.makedirs(output_dir, exist_ok=True)

# Kullanılacak gri görüntü
image_path = r"C:\Users\Casper\OneDrive\Masaüstü\Üniversiteye Dair Her Şey\3.Sınıf\bahar dönemi\Görüntü İşleme\Projeler\Renklendirme\dataset\grayscale\image_1.jpg"
img = Image.open(image_path).convert("L")

# Dönüşüm tanımı
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
x = transform(img).unsqueeze(0).to(device)

# Tüm .pth dosyaları üzerinde dön
for model_file in os.listdir(model_dir):
    if model_file.endswith(".pth") and "generator" in model_file.lower():
        model_path = os.path.join(model_dir, model_file)

        try:
            # Modeli yükle
            G = Generator().to(device)
            G.load_state_dict(torch.load(model_path, map_location=device))
            G.eval()

            # Tahmin
            with torch.no_grad():
                y = G(x).squeeze().cpu() * 0.5 + 0.5

            # Çıktıyı kaydet
            out = transforms.ToPILImage()(y.clamp(0, 1))
            model_name = os.path.splitext(model_file)[0]
            save_path = os.path.join(output_dir, f"{model_name}.jpg")
            out.save(save_path)
            print(f"{model_name} çıktısı kaydedildi.")
        
        except Exception as e:
            print(f"{model_file} modeli yüklenemedi: {e}")
