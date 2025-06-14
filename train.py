import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from models.generator import Generator
from models.discriminator import Discriminator
from utils import GrayscaleColorDataset
import matplotlib.pyplot as plt
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator().to(device)
D = Discriminator().to(device)
opt_G = torch.optim.Adam(G.parameters(), lr=1e-4)
opt_D = torch.optim.Adam(D.parameters(), lr=5e-5)

dataset = GrayscaleColorDataset()
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=50, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=50, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=50, shuffle=False)

g_losses, d_losses = [], []
g_val_losses, d_val_losses = [], []
start_epoch = 1

os.makedirs("saved_models2", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("results3", exist_ok=True)  

checkpoint_path = "saved_models2/checkpoint.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    G.load_state_dict(checkpoint["G"])
    D.load_state_dict(checkpoint["D"])
    opt_G.load_state_dict(checkpoint["opt_G"])
    opt_D.load_state_dict(checkpoint["opt_D"])
    start_epoch = checkpoint["epoch"] + 1
    g_losses = checkpoint["g_losses"]
    d_losses = checkpoint["d_losses"]
    g_val_losses = checkpoint["g_val_losses"]
    d_val_losses = checkpoint["d_val_losses"]
    print(f"\nCheckpoint bulundu. Eğitim kaldığı yerden devam ediyor → Epoch {start_epoch}\n")
else:
    print("\nCheckpoint bulunamadı. Eğitim sıfırdan başlıyor.\n")

# En iyi Val G'yi takip et
best_val_g_loss = float("inf")

for epoch in range(start_epoch, 238):  
    G.train()
    D.train()
    g_total, d_total = 0.0, 0.0

    for gray, real in train_loader:
        gray, real = gray.to(device), real.to(device)
        fake = G(gray)

        # --- Discriminator adımı ---
        D.zero_grad()
        pred_real = D(gray, real)
        real_labels = torch.full((pred_real.size(0), 1), 0.9, device=device)
        pred_fake = D(gray, fake.detach())
        fake_labels = torch.zeros_like(real_labels)
        d_loss_real = F.binary_cross_entropy(pred_real, real_labels)
        d_loss_fake = F.binary_cross_entropy(pred_fake, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        opt_D.step()

        # --- Generator adımı ---
        G.zero_grad()
        pred_fake = D(gray, fake)
        adv_loss = F.binary_cross_entropy(pred_fake, real_labels)
        l1_loss = F.l1_loss(fake, real)
        g_loss = adv_loss + 100 * l1_loss
        g_loss.backward()
        opt_G.step()

        d_total += d_loss.item()
        g_total += g_loss.item()

    g_loss_epoch = g_total / len(train_loader)
    d_loss_epoch = d_total / len(train_loader)
    g_losses.append(g_loss_epoch)
    d_losses.append(d_loss_epoch)

    # Validation
    G.eval()
    D.eval()
    g_val_total, d_val_total = 0.0, 0.0

    with torch.no_grad():
        for gray, real in val_loader:
            gray, real = gray.to(device), real.to(device)
            fake = G(gray)

            pred_real = D(gray, real)
            real_labels = torch.full((pred_real.size(0), 1), 0.9, device=device)
            pred_fake = D(gray, fake)
            fake_labels = torch.zeros_like(real_labels)

            d_loss_real = F.binary_cross_entropy(pred_real, real_labels)
            d_loss_fake = F.binary_cross_entropy(pred_fake, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) * 0.5

            adv_loss = F.binary_cross_entropy(pred_fake, real_labels)
            l1_loss = F.l1_loss(fake, real)
            g_loss = adv_loss + 100 * l1_loss

            d_val_total += d_loss.item()
            g_val_total += g_loss.item()

    g_val_loss = g_val_total / len(val_loader)
    d_val_loss = d_val_total / len(val_loader)
    g_val_losses.append(g_val_loss)
    d_val_losses.append(d_val_loss)

    print(f"[{epoch}/100] Train G: {g_loss_epoch:.4f} | Train D: {d_loss_epoch:.4f} || Val G: {g_val_loss:.4f} | Val D: {d_val_loss:.4f}")

    # ✅ En iyi Val G kontrolü
    if g_val_loss < best_val_g_loss:
        best_val_g_loss = g_val_loss
        torch.save(G.state_dict(), "results3/best_generator.pth")
        torch.save(D.state_dict(), "results3/best_discriminator.pth")
        print(f"💾 En iyi Val G bulundu! Model kaydedildi → Val G: {best_val_g_loss:.4f}")

    torch.save(G.state_dict(), f"saved_models2/generator_epoch{epoch}.pth")
    torch.save(D.state_dict(), f"saved_models2/discriminator_epoch{epoch}.pth")

    torch.save({
        "epoch": epoch,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
        "g_losses": g_losses,
        "d_losses": d_losses,
        "g_val_losses": g_val_losses,
        "d_val_losses": d_val_losses
    }, checkpoint_path)

# Final model kaydı
torch.save(G.state_dict(), "saved_models2/generator_final.pth")
torch.save(D.state_dict(), "saved_models2/discriminator_final.pth")

# Test değerlendirmesi
G.eval()
D.eval()
g_test_total, d_test_total = 0.0, 0.0

with torch.no_grad():
    for gray, real in test_loader:
        gray, real = gray.to(device), real.to(device)
        fake = G(gray)

        pred_real = D(gray, real)
        real_labels = torch.full((pred_real.size(0), 1), 0.9, device=device)
        pred_fake = D(gray, fake)
        fake_labels = torch.zeros_like(real_labels)

        d_loss_real = F.binary_cross_entropy(pred_real, real_labels)
        d_loss_fake = F.binary_cross_entropy(pred_fake, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) * 0.5

        adv_loss = F.binary_cross_entropy(pred_fake, real_labels)
        l1_loss = F.l1_loss(fake, real)
        g_loss = adv_loss + 100 * l1_loss

        d_test_total += d_loss.item()
        g_test_total += g_loss.item()

g_test_loss = g_test_total / len(test_loader)
d_test_loss = d_test_total / len(test_loader)
print(f"\nTest Sonuçları → Generator: {g_test_loss:.4f} | Discriminator: {d_test_loss:.4f}")

# Kayıpların grafiği
plt.figure()
plt.plot(g_losses, label="Train G")
plt.plot(g_val_losses, label="Val G")
plt.plot(d_losses, label="Train D")
plt.plot(d_val_losses, label="Val D")
plt.legend()
plt.title("Train/Validation Loss")
plt.savefig("results/loss.png")



import torch
import torchvision.utils as vutils
from torchmetrics.functional import structural_similarity_index_measure as ssim
import torch.nn.functional as F

# -- 1. İstediğin epoch'lardaki kayıpları yazdırma --
check_epochs = [19, 182, 238]
print("\nEpoch | Train G Loss | Val G Loss")
print("-" * 30)
for e in check_epochs:
    if e - 1 < len(g_losses) and e - 1 < len(g_val_losses):
        print(f"{e:<5} | {g_losses[e-1]:.4f}     | {g_val_losses[e-1]:.4f}")
    else:
        print(f"{e:<5} | Veri yok (epoch aralığı dışı)")

# -- 2. Test setinden örnek görselleri kaydetme --
G.eval()
with torch.no_grad():
    for i, (gray, real) in enumerate(test_loader):
        gray = gray.to(device)
        fake = G(gray)
        # İlk batch'ten ilk 4 örneği kaydet
        if i == 0:
            # Gri giriş (input)
            vutils.save_image(gray[:3], "results/grayscale_sample.png", normalize=True)
            # Üretilen renkli çıktı
            vutils.save_image(fake[:3], "results/colored_sample.png", normalize=True)
            break
print("\nBelirtilen epoch'lar için kayıplar ve test örnek görselleri results klasörüne kaydedildi.")

# -- 3. Test seti için MAE, MSE, SSIM metriklerini hesaplama ve kaydetme --
mae_total = 0.0
mse_total = 0.0
ssim_total = 0.0
num_batches = 0

with torch.no_grad():
    for gray, real in test_loader:
        gray, real = gray.to(device), real.to(device)
        fake = G(gray)

        mae_total += F.l1_loss(fake, real).item()
        mse_total += F.mse_loss(fake, real).item()
        ssim_total += ssim(fake, real, data_range=1.0).item()

        num_batches += 1

mae_avg = mae_total / num_batches
mse_avg = mse_total / num_batches
ssim_avg = ssim_total / num_batches

print("\n Test Metrikleri:")
print(f"MAE: {mae_avg:.4f}")
print(f"MSE: {mse_avg:.4f}")
print(f"SSIM: {ssim_avg:.4f}")

# Metrikleri dosyaya kaydetme
with open("results/results.txt", "w") as f:
    f.write("Test Seti Metrikleri:\n ")
    f.write(f"MAE: {mae_avg:.4f}\n")
    f.write(f"MSE: {mse_avg:.4f}\n")
    f.write(f"SSIM: {ssim_avg:.4f}\n")




"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from models.generator import Generator
from models.discriminator import Discriminator
from utils import GrayscaleColorDataset
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator().to(device)
D = Discriminator().to(device)
opt_G = torch.optim.Adam(G.parameters(), lr=1e-4)
opt_D = torch.optim.Adam(D.parameters(), lr=5e-5)

dataset = GrayscaleColorDataset()
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=50, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=50, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=50, shuffle=False)

g_losses, d_losses = [], []
g_val_losses, d_val_losses = [], []
start_epoch = 1

os.makedirs("saved_models2", exist_ok=True)
os.makedirs("results", exist_ok=True)

checkpoint_path = "saved_models2/checkpoint.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    G.load_state_dict(checkpoint["G"])
    D.load_state_dict(checkpoint["D"])
    opt_G.load_state_dict(checkpoint["opt_G"])
    opt_D.load_state_dict(checkpoint["opt_D"])
    start_epoch = checkpoint["epoch"] + 1
    g_losses = checkpoint["g_losses"]
    d_losses = checkpoint["d_losses"]
    g_val_losses = checkpoint["g_val_losses"]
    d_val_losses = checkpoint["d_val_losses"]
    print(f"\n✅ Checkpoint bulundu. Eğitim kaldığı yerden devam ediyor → Epoch {start_epoch}\n")
else:
    print("\n🚀 Checkpoint bulunamadı. Eğitim sıfırdan başlıyor.\n")

for epoch in range(start_epoch, 150):  
    G.train()
    D.train()
    g_total, d_total = 0.0, 0.0

    for gray, real in train_loader:
        gray, real = gray.to(device), real.to(device)
        fake = G(gray)

        # --- Discriminator adımı ---
        D.zero_grad()
        pred_real = D(gray, real)
        real_labels = torch.full((pred_real.size(0), 1), 0.9, device=device)
        pred_fake = D(gray, fake.detach())
        fake_labels = torch.zeros_like(real_labels)
        d_loss_real = F.binary_cross_entropy(pred_real, real_labels)
        d_loss_fake = F.binary_cross_entropy(pred_fake, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        opt_D.step()

        # --- Generator adımı ---
        G.zero_grad()
        pred_fake = D(gray, fake)
        adv_loss = F.binary_cross_entropy(pred_fake, real_labels)  # gerçek label (0.9) ile kandırmaya çalışıyor
        l1_loss = F.l1_loss(fake, real)
        g_loss = adv_loss + 100 * l1_loss
        g_loss.backward()
        opt_G.step()

        d_total += d_loss.item()
        g_total += g_loss.item()

    g_loss_epoch = g_total / len(train_loader)
    d_loss_epoch = d_total / len(train_loader)
    g_losses.append(g_loss_epoch)
    d_losses.append(d_loss_epoch)

    # Validation
    G.eval()
    D.eval()
    g_val_total, d_val_total = 0.0, 0.0

    with torch.no_grad():
        for gray, real in val_loader:
            gray, real = gray.to(device), real.to(device)
            fake = G(gray)

            pred_real = D(gray, real)
            real_labels = torch.full((pred_real.size(0), 1), 0.9, device=device)
            pred_fake = D(gray, fake)
            fake_labels = torch.zeros_like(real_labels)

            d_loss_real = F.binary_cross_entropy(pred_real, real_labels)
            d_loss_fake = F.binary_cross_entropy(pred_fake, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) * 0.5

            adv_loss = F.binary_cross_entropy(pred_fake, real_labels)
            l1_loss = F.l1_loss(fake, real)
            g_loss = adv_loss + 100 * l1_loss

            d_val_total += d_loss.item()
            g_val_total += g_loss.item()

    g_val_loss = g_val_total / len(val_loader)
    d_val_loss = d_val_total / len(val_loader)
    g_val_losses.append(g_val_loss)
    d_val_losses.append(d_val_loss)

    print(f"[{epoch}/100] Train G: {g_loss_epoch:.4f} | Train D: {d_loss_epoch:.4f} || Val G: {g_val_loss:.4f} | Val D: {d_val_loss:.4f}")

    torch.save(G.state_dict(), f"saved_models2/generator_epoch{epoch}.pth")
    torch.save(D.state_dict(), f"saved_models2/discriminator_epoch{epoch}.pth")

    torch.save({
        "epoch": epoch,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
        "g_losses": g_losses,
        "d_losses": d_losses,
        "g_val_losses": g_val_losses,
        "d_val_losses": d_val_losses
    }, checkpoint_path)

# Final model kaydı
torch.save(G.state_dict(), "saved_models2/generator_final.pth")
torch.save(D.state_dict(), "saved_models2/discriminator_final.pth")

# Test değerlendirmesi
G.eval()
D.eval()
g_test_total, d_test_total = 0.0, 0.0

with torch.no_grad():
    for gray, real in test_loader:
        gray, real = gray.to(device), real.to(device)
        fake = G(gray)

        pred_real = D(gray, real)
        real_labels = torch.full((pred_real.size(0), 1), 0.9, device=device)
        pred_fake = D(gray, fake)
        fake_labels = torch.zeros_like(real_labels)

        d_loss_real = F.binary_cross_entropy(pred_real, real_labels)
        d_loss_fake = F.binary_cross_entropy(pred_fake, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) * 0.5

        adv_loss = F.binary_cross_entropy(pred_fake, real_labels)
        l1_loss = F.l1_loss(fake, real)
        g_loss = adv_loss + 100 * l1_loss

        d_test_total += d_loss.item()
        g_test_total += g_loss.item()

g_test_loss = g_test_total / len(test_loader)
d_test_loss = d_test_total / len(test_loader)
print(f"\n📊 Test Sonuçları → Generator: {g_test_loss:.4f} | Discriminator: {d_test_loss:.4f}")

# Kayıpların grafiği
plt.figure()
plt.plot(g_losses, label="Train G")
plt.plot(g_val_losses, label="Val G")
plt.plot(d_losses, label="Train D")
plt.plot(d_val_losses, label="Val D")
plt.legend()
plt.title("Train/Validation Loss")
plt.savefig("results/loss.png")
"""