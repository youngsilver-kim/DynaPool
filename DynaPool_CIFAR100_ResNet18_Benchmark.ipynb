# DynaPool Benchmark (CIFAR-100, ResNet-18)

- Pools: Average / Max / GeM / Attention / DynaPool (gated mixture)
- Regime: 300 epochs, batch 128, Adam (lr=1e-3, wd=1e-4), cosine schedule
- Augmentation: random crop (4px), horizontal flip; normalized CIFAR-100
- Report: best top-1 accuracy (test), wall-clock training time (minutes)
- Note: GeM uses clamped p ∈ [1e-3, 10]; DynaPool uses GAP→MLP→softmax gating


# =========================
# CIFAR-100 ResNet-18: Avg / Max / GeM / Attention / DynaPool
# One-cell benchmark for 300 epochs, prints a summary table
# =========================
!pip -q install tqdm

import os, math, time, random
from typing import Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as T
from torchvision.models import resnet18
from tqdm import tqdm

# ---- Global config ----
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT     = "data"
OUTPUT_DIR    = "checkpoints_benchmark"
EPOCHS        = 300
BATCH_SIZE    = 128
BASE_LR       = 1e-3
WEIGHT_DECAY  = 1e-4
LABEL_SMOOTH  = 0.0
NUM_WORKERS   = 2
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed=42):
    """Lightweight reproducibility; cuDNN benchmark stays on for speed."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
set_seed(42)

# ---- Pooling modules ----
class GeMPooling(nn.Module):
    """Generalized Mean Pooling with clamped p for numerical stability."""
    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init))
        self.eps = eps
    def forward(self, x):
        p = torch.clamp(self.p, min=1e-3, max=10.0)
        x = torch.clamp(x, min=self.eps)
        x = x.pow(p)
        x = F.avg_pool2d(x, kernel_size=(x.size(-2), x.size(-1)))
        x = x.pow(1.0/p)
        return x.squeeze(-1).squeeze(-1)

class AvgPooling(nn.Module):
    def forward(self, x): return F.adaptive_avg_pool2d(x, 1).flatten(1)

class MaxPooling(nn.Module):
    def forward(self, x): return F.adaptive_max_pool2d(x, 1).flatten(1)

class AttentionPooling(nn.Module):
    """1×1 conv attention → softmax over spatial positions → weighted sum."""
    def __init__(self, in_channels:int):
        super().__init__()
        self.score = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
    def forward(self, x):
        N, C, H, W = x.shape
        s = self.score(x).view(N, 1, H*W)
        att = F.softmax(s, dim=-1)
        x_flat = x.view(N, C, H*W)
        f = torch.bmm(x_flat, att.transpose(1,2)).squeeze(-1)
        return f

# ---- Heads ----
class SinglePoolHead(nn.Module):
    """Avg/Max/GeM/Attention → FC classifier."""
    def __init__(self, in_channels:int, num_classes:int, kind:str):
        super().__init__()
        self.kind = kind
        if kind == "avg":
            self.pool = AvgPooling()
        elif kind == "max":
            self.pool = MaxPooling()
        elif kind == "gem":
            self.pool = GeMPooling(p_init=3.0)
        elif kind == "att":
            self.pool = AttentionPooling(in_channels)
        else:
            raise ValueError(kind)
        self.fc = nn.Linear(in_channels, num_classes)
    def forward(self, fmap):
        f = self.pool(fmap)
        out = self.fc(f)
        return out

class DynaPoolHead(nn.Module):
    """Avg/Max/GeM/Attention branches + GAP→MLP gating (softmax) → FC."""
    def __init__(self, in_channels:int, num_classes:int, hidden=256, dropout=0.1):
        super().__init__()
        self.avg = AvgPooling()
        self.max = MaxPooling()
        self.gem = GeMPooling(p_init=3.0)
        self.att = AttentionPooling(in_channels)
        self.gap = AvgPooling()
        self.gate = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 4)
        )
        self.fc = nn.Linear(in_channels, num_classes)
    def forward(self, fmap):
        f_avg = self.avg(fmap)
        f_max = self.max(fmap)
        f_gem = self.gem(fmap)
        f_att = self.att(fmap)
        alpha = F.softmax(self.gate(self.gap(fmap)), dim=-1)  # (N, 4)
        f = (alpha[:,0:1]*f_avg +
             alpha[:,1:2]*f_max +
             alpha[:,2:3]*f_gem +
             alpha[:,3:4]*f_att)
        out = self.fc(f)
        return out, alpha

# ---- Backbone ----
class ResNet18Backbone(nn.Module):
    """ResNet-18 up to layer4; returns final feature map (C=512)."""
    def __init__(self):
        super().__init__()
        base = resnet18(weights=None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.out_ch = 512
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return x

# ---- Data ----
def get_dataloaders(root, batch, num_workers=2):
    """CIFAR-100 with standard mean/std; crop+flip for train."""
    mean, std = (0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761)
    train_tf = T.Compose([T.RandomCrop(32,padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(mean,std)])
    test_tf  = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
    train_ds = torchvision.datasets.CIFAR100(root=root, train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=test_tf)
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_ld, test_ld

# ---- Train/Eval ----
def accuracy(logits, target):
    return (logits.argmax(1) == target).float().mean().item()

def cosine_lr(optimizer, base_lr, epoch, total_epochs, warmup=5):
    """Cosine with linear warmup (per-epoch update)."""
    if epoch < warmup: lr = base_lr*(epoch+1)/warmup
    else:
        progress = (epoch-warmup)/max(1,(total_epochs-warmup))
        lr = 0.5*base_lr*(1+math.cos(math.pi*progress))
    for pg in optimizer.param_groups: pg['lr'] = lr
    return lr

def train_epoch(model, loader, optimizer, scaler, device, epoch, total_epochs, base_lr, label_smoothing=0.0):
    model.train()
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    run_loss=0; run_acc=0; n=0
    lr = cosine_lr(optimizer, base_lr, epoch, total_epochs)
    pbar = tqdm(loader, desc=f"Train {epoch+1}/{total_epochs} (lr={lr:.5f})", ncols=100)
    for x,y in pbar:
        x,y = x.to(device,non_blocking=True), y.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = ce(logits, y)
        if device.type=='cuda':
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        bs = x.size(0)
        run_loss += loss.item()*bs; run_acc += accuracy(logits,y)*bs; n += bs
        pbar.set_postfix(loss=f"{run_loss/n:.4f}", acc=f"{run_acc/n:.4f}")
    return run_loss/n, run_acc/n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ce=nn.CrossEntropyLoss()
    total_loss=0; total_acc=0; n=0
    alpha_sum = torch.zeros(4, device=device)
    for x,y in tqdm(loader, desc="Eval", ncols=100):
        x,y = x.to(device), y.to(device)
        out = model(x)
        if isinstance(out, tuple):
            logits, alpha = out
            alpha_sum += alpha.mean(0)
        else:
            logits = out
        loss = ce(logits, y)
        bs = x.size(0)
        total_loss += loss.item()*bs; total_acc += accuracy(logits,y)*bs; n += bs
    avg_alpha = (alpha_sum/len(loader)).detach().cpu().tolist()
    return total_loss/n, total_acc/n, avg_alpha

# ---- Runner for each method ----
def run_method(method:str):
    backbone = ResNet18Backbone().to(DEVICE)
    if method == "dyna":
        head = DynaPoolHead(backbone.out_ch, 100).to(DEVICE)
        class Model(nn.Module):
            def __init__(self, bb, hd): super().__init__(); self.bb=bb; self.hd=hd
            def forward(self, x): fmap = self.bb(x); return self.hd(fmap)
        model = Model(backbone, head).to(DEVICE)
    else:
        head = SinglePoolHead(backbone.out_ch, 100, kind=method).to(DEVICE)
        class Model(nn.Module):
            def __init__(self, bb, hd): super().__init__(); self.bb=bb; self.hd=hd
            def forward(self, x): fmap = self.bb(x); return self.hd(fmap)
        model = Model(backbone, head).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type=='cuda'))

    best_acc = 0.0
    start_t = time.time()
    for ep in range(EPOCHS):
        train_epoch(model, train_loader, opt, scaler, DEVICE, ep, EPOCHS, BASE_LR, LABEL_SMOOTH)
        _, te_acc, _ = evaluate(model, test_loader, DEVICE)
        if te_acc > best_acc: best_acc = te_acc
    minutes = (time.time()-start_t)/60.0

    # Report GeM p or adaptivity flag
    gem_p = "-"
    adapt = "✗"
    if method == "gem":
        gem_p = f"{float(head.pool.p.data):.4f}"
        adapt = "Global"
    if method == "dyna":
        gem_p = "Adaptive"
        adapt = "✓"
    elif method == "att":
        adapt = "Local"

    return best_acc, minutes, gem_p, adapt

# ---- Prepare data once ----
train_loader, test_loader = get_dataloaders(DATA_ROOT, BATCH_SIZE, NUM_WORKERS)

# ---- Run all methods ----
methods = [("Average Pooling","avg"),
           ("Max Pooling","max"),
           ("GeM Pooling","gem"),
           ("Attention Pooling","att"),
           ("DynaPool (Ours)","dyna")]

summary = []
for name, key in methods:
    print(f"\n==== Running: {name} ({key}) ====")
    acc, mins, gem_p, adapt = run_method(key)
    summary.append((name, acc, mins, gem_p, adapt))

# ---- Print summary table ----
def fmt(x): return f"{x:.4f}"
header = ["Pooling Method","Test Accuracy","Training Time (min)","GeM p","Adaptivity"]
print("\n" + "="*72)
print("Table 1. Summary of Experimental Results")
print("-"*72)
print(f"{header[0]:<20} {header[1]:>14} {header[2]:>20} {header[3]:>10} {header[4]:>12}")
for row in summary:
    name, acc, mins, gp, ad = row
    acc_s = fmt(acc) if isinstance(acc,(float,int)) else str(acc)
    mins_s = f"{mins:.2f}"
    print(f"{name:<20} {acc_s:>14} {mins_s:>20} {gp:>10} {ad:>12}")
print("="*72)
