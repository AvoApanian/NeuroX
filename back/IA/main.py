import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dotenv import load_dotenv
import psycopg2
from PIL import Image, ImageFilter, ImageEnhance
from io import BytesIO
import os
import random

load_dotenv()

DBHOST = os.getenv("dbHost")
DBNAME = os.getenv("dbName")
DBUSER = os.getenv("dbUser")
DBPORT = os.getenv("dbPort")
DBPASSWORD = os.getenv("dbPassword")
DBTABLE = os.getenv("dbTable", "iaimg")

IMGSIZE = (256, 256)
BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

conn = psycopg2.connect(
    host=DBHOST, database=DBNAME,
    user=DBUSER, password=DBPASSWORD, port=DBPORT
)
cur = conn.cursor()
cur.execute(f"SELECT high FROM {DBTABLE}")
ROWS = cur.fetchall()
cur.close()
conn.close()

print(f"Images chargées: {len(ROWS)}")

class SharpDataset(Dataset):
    def __init__(self, rows, noise_level=0.005):
        self.images = [row[0] for row in rows]
        self.noise_level = noise_level

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        high_img = Image.open(BytesIO(self.images[idx])).convert("RGB")
        high_img = high_img.resize(IMGSIZE, Image.BICUBIC)
        if random.random() < 0.5:
            high_img = high_img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.3:
            high_img = high_img.rotate(random.choice([90, 180, 270]))

        scale = random.uniform(1.5, 2.5)  
        small = high_img.resize(
            (int(IMGSIZE[0]/scale), int(IMGSIZE[1]/scale)),
            Image.LANCZOS  
        )
        low_img = small.resize(IMGSIZE, Image.LANCZOS)
        
        low_img = low_img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5)))

        buf = BytesIO()
        low_img.save(buf, format="JPEG", quality=random.randint(70, 90))
        buf.seek(0)
        low_img = Image.open(buf)

        high = torch.from_numpy(np.array(high_img)).float().permute(2,0,1) / 255.0
        low = torch.from_numpy(np.array(low_img)).float().permute(2,0,1) / 255.0

        noise = torch.randn_like(low) * self.noise_level
        low = torch.clamp(low + noise, 0, 1)

        return low, high

dataset = SharpDataset(ROWS)
loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=True
)

#MODEL ANTI-PIXELISATION 
class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch//reduction, ch, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg(x))

class RCAB(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),  
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch)
        )
        self.ca = ChannelAttention(ch)

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        return x + res * 0.1  

class ResidualGroup(nn.Module):
    def __init__(self, ch, n_blocks=4):
        super().__init__()
        self.blocks = nn.Sequential(*[RCAB(ch) for _ in range(n_blocks)])
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        res = self.blocks(x)
        res = self.conv(res)
        return x + res * 0.1

class SharpSRModel(nn.Module):
    def __init__(self):
        super().__init__()
        nf = 64  
        
        self.head = nn.Conv2d(3, nf, 3, padding=1)
        
        self.body = nn.Sequential(
            ResidualGroup(nf, n_blocks=4),
            ResidualGroup(nf, n_blocks=4)
        )
        
        self.body_tail = nn.Conv2d(nf, nf, 3, padding=1)
        
        self.tail = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf//2, nf//4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf//4, 3, 3, padding=1)
        )

    def forward(self, x):
        feat = self.head(x)
        
        res = self.body(feat)
        res = self.body_tail(res)
        res += feat
        
        residual = self.tail(res) * 0.25  
        
        return torch.clamp(x + residual, 0, 1)

model = SharpSRModel().to(DEVICE)
print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")

mse = nn.MSELoss()
l1 = nn.L1Loss()

def grad_loss(pred, target):
    """Gradient loss pour détails"""
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)

def smooth_loss(pred):
    dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    return torch.mean(dx) + torch.mean(dy)

def texture_loss(pred, target):
    pred_patches = F.unfold(pred, kernel_size=3, padding=1)
    target_patches = F.unfold(target, kernel_size=3, padding=1)
    
    pred_var = torch.var(pred_patches, dim=1)
    target_var = torch.var(target_patches, dim=1)
    
    return F.l1_loss(pred_var, target_var)

def sharp_loss(pred, target):
    return (
        0.20 * mse(pred, target) +         
        0.25 * l1(pred, target) +          
        0.35 * grad_loss(pred, target) +
        0.10 * texture_loss(pred, target) + 
        0.10 * smooth_loss(pred)           
    )

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

best_loss = float("inf")
print("Train")

noise_schedule = [0.005] * 30 + [0.002] * 15 + [0.0] * 5

for epoch in range(EPOCHS):
    model.train()
    total = 0
    
    current_noise = noise_schedule[min(epoch, len(noise_schedule)-1)]
    dataset.noise_level = current_noise

    for low, high in loader:
        low, high = low.to(DEVICE), high.to(DEVICE)

        optimizer.zero_grad()
        pred = model(low)
        loss = sharp_loss(pred, high)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total += loss.item()
        
        del pred, loss
        
    avg = total / len(loader)
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg:.6f} | LR: {lr:.2e} | Noise: {current_noise:.4f}")

    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), "best_sharp_model.pth")
        print("  Best model saved")

    scheduler.step()
    
    if epoch % 5 == 0:
        torch.cuda.empty_cache()

model.eval()
dummy = torch.randn(1, 3, 256, 256).to(DEVICE)

torch.onnx.export(
    model, dummy, "IaModel_Sharp.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)

print("\nExport ONNX OK")
print(f"Best loss: {best_loss:.6f}")