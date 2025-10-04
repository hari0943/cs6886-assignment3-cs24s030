#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobileNetV2 (from scratch) on CIFAR-10
- Correct inverted residual architecture (no torchvision import)
- 3x3/1 stem tailored for 32x32 CIFAR inputs
- SGD + Cosine LR with warmup
- AMP mixed precision
- Best-checkpoint saving & resume
"""

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Iterable, Tuple
#import wandb
#wandb.init(project="mobilenetv2-cifar10")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----------------------------
# Repro & small utilities
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True  # good for convnets

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k * (100.0 / target.size(0))).item())
        return res

# ----------------------------
# MobileNetV2 building blocks
# ----------------------------
def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )

class InvertedResidual(nn.Module):
    """
    MobileNetV2 inverted residual with linear bottleneck.
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise (expand)
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        # 3x3 depthwise
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        # 1x1 pointwise (project, linear)
        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

@dataclass
class MV2Config:
    num_classes: int = 10
    width_mult: float = 1.0
    round_nearest: int = 8  # channel rounding

class MobileNetV2(nn.Module):
    """
    MobileNetV2 backbone adapted for CIFAR-10 (32x32):
      - stem is 3x3/1 (not 3x3/2) so we don't downsample too aggressively early
      - last layer to num_classes
    """
    def __init__(self, cfg: MV2Config):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        width_mult = cfg.width_mult
        round_nearest = cfg.round_nearest

        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        # t, c, n, s for each stage (from the paper), but tweak stem stride for CIFAR
        inverted_residual_setting = [
            # expand, out_c, repeats, stride
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # keep stride=1 (CIFAR small); if you want more downsample, set first to 2
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # stem: 3x3/1 for CIFAR-10
        features.append(conv_3x3_bn(3, input_channel, 1))

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # last 1x1 conv
        features.append(conv_1x1_bn(input_channel, last_channel))
        self.features = nn.Sequential(*features)

        # classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, cfg.num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

# ----------------------------
# Data
# ----------------------------
def build_cifar10_loaders(data_dir: str, batch_size: int, workers: int) -> Tuple[DataLoader, DataLoader]:
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

    imgs = [item[0] for item in cifar_trainset] # item[0] and item[1] are image and its label
    imgs = torch.stack(imgs, dim=0).numpy()

    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:,0,:,:].mean()
    mean_g = imgs[:,1,:,:].mean()
    mean_b = imgs[:,2,:,:].mean()

    # calculate std over each channel (r,g,b)
    std_r = imgs[:,0,:,:].std()
    std_g = imgs[:,1,:,:].std()
    std_b = imgs[:,2,:,:].std()
    mean = (mean_r, mean_g, mean_b)
    std = (std_r,std_g,std_b)
    print(mean,std)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=True)
    return train_loader, test_loader

# ----------------------------
# Optimizer & Schedulers
# ----------------------------
class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # linear warmup
            warmup_factor = float(self.last_epoch + 1) / float(max(1, self.warmup_epochs))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        # cosine
        progress = (self.last_epoch - self.warmup_epochs) / float(max(1, self.max_epochs - self.warmup_epochs))
        return [base_lr * 0.5 * (1.0 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]

# ----------------------------
# Train / Evaluate
# ----------------------------
def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    scaler: torch.cuda.amp.GradScaler,
                    epoch: int,
                    log_interval: int = 100):
    model.train()
    loss_meter = 0.0
    top1_meter = 0.0

    for it, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',enabled=scaler is not None):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        acc1, = accuracy(logits, targets, topk=(1,))
        loss_meter += loss.item()
        top1_meter += acc1

        if (it + 1) % log_interval == 0:
            print(f"Epoch {epoch} | Iter {it+1}/{len(loader)} | "
                  f"Loss {loss_meter/(it+1):.4f} | Acc@1 {top1_meter/(it+1):.2f}%")

    return loss_meter / len(loader), top1_meter / len(loader)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    loss_meter = 0.0
    top1_meter = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        acc1, = accuracy(logits, targets, topk=(1,))
        loss_meter += loss.item()
        top1_meter += acc1

    return loss_meter / len(loader), top1_meter / len(loader)

# ----------------------------
# Checkpoint utils
# ----------------------------
def save_checkpoint(state: dict, is_best: bool, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "last.pth")
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(out_dir, "best.pth"))

def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_acc1 = ckpt.get("best_acc1", 0.0)
    print(f"Loaded checkpoint from {path} (epoch {start_epoch-1}, best_acc1 {best_acc1:.2f}%)")
    return start_epoch, best_acc1

# ----------------------------
# Main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MobileNetV2 from scratch on CIFAR-10")
    p.add_argument("--data", type=str, default="./data", help="dataset root")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--width-mult", type=float, default=1.0, help="MobileNetV2 width multiplier")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--out", type=str, default="./runs/mobilenetv2_cifar10")
    p.add_argument("--resume", type=str, default="", help="path to checkpoint to resume from")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = build_cifar10_loaders(args.data, args.batch_size, args.workers)

    # Model
    cfg = MV2Config(num_classes=10, width_mult=args.width_mult)
    model = MobileNetV2(cfg).to(device)

    # Optimizer / Scheduler / AMP
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=True
    )
    scheduler = CosineWithWarmup(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs)
    scaler = None if args.no_amp or device.type != "cuda" else torch.amp.GradScaler('cuda')

    # Optionally resume
    start_epoch = 0
    best_acc1 = 0.0
    if args.resume:
        start_epoch, best_acc1 = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)

    # Train
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        train_loss, train_acc1 = train_one_epoch(model, train_loader, optimizer, device, scaler, epoch+1)
        val_loss, val_acc1 = evaluate(model, test_loader, device)
        """        
        wandb.log({
        "epoch": epoch+1,
        "train/loss": train_loss, "train/acc1": train_acc1,
        "val/loss": val_loss, "val/acc1": val_acc1,
        "lr": optimizer.param_groups[0]["lr"]
        })"""
        scheduler.step()

        print(f"[Train] loss: {train_loss:.4f} | acc@1: {train_acc1:.2f}%")
        print(f"[Val]   loss: {val_loss:.4f} | acc@1: {val_acc1:.2f}% | best: {best_acc1:.2f}%")

        is_best = val_acc1 > best_acc1
        best_acc1 = max(best_acc1, val_acc1)
        save_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_acc1": best_acc1,
                "args": vars(args),
            },
            is_best=is_best,
            out_dir=args.out,
        )

    print(f"\nTraining complete. Best Val Acc@1: {best_acc1:.2f}%")
    print(f"Checkpoints saved under: {args.out}")

if __name__ == "__main__":
    main()
