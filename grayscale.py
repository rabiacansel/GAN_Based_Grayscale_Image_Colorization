from PIL import Image
import os

base_dir = os.getcwd()  
color_images = os.path.join(base_dir, "dataset", "rgb_image")
gray_images = os.path.join(base_dir, "dataset", "grayscale")
os.makedirs(gray_images, exist_ok=True)

for fname in os.listdir(color_images):
    if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(color_images, fname)
        img = Image.open(img_path).convert("RGB")  # Renk uyumluluğu
        img = img.resize((1024, 1024))             # Tek tip boyut
        gray = img.convert('L')                    # Griye çevir
        gray.save(os.path.join(gray_images, fname)) 
