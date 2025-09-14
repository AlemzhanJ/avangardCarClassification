#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    img_size: int = 384
    batch_size: int = 16
    epochs: int = 5
    lr: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    model: str = "resnet50"
    use_pretrained: bool = True
    freeze_epochs: int = 3
    early_stop_patience: int = 5
    num_workers: int = 4
    mixed_precision: bool = True
    balance: str = "sampler"  # "sampler", "weights", "none"
    seed: int = 42
    save_last_k: int = 3
    device: str = "auto"  # auto, cuda, mps, cpu


def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(3/4, 4/3)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.RandomRotation(degrees=10),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def make_datasets(data_dir: str, img_size: int):
    train_tf, eval_tf = build_transforms(img_size)
    train_ds = ImageFolder(os.path.join(data_dir, 'train'), transform=train_tf)
    val_ds = ImageFolder(os.path.join(data_dir, 'valid'), transform=eval_tf)
    test_path = os.path.join(data_dir, 'test')
    test_ds = ImageFolder(test_path, transform=eval_tf) if os.path.isdir(test_path) else None
    return train_ds, val_ds, test_ds


def class_counts(dataset: ImageFolder) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for _, y in dataset.samples:
        counts[y] = counts.get(y, 0) + 1
    return counts


def resolve_device(requested: str) -> torch.device:
    if requested == 'cuda':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested == 'mps':
        return torch.device('mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu')
    if requested == 'cpu':
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def make_dataloaders(cfg: TrainConfig):
    train_ds, val_ds, test_ds = make_datasets(cfg.data_dir, cfg.img_size)

    counts = class_counts(train_ds)
    classes = sorted(counts.keys())
    num_classes = len(classes)
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}

    total = sum(counts.values())
    class_weights = torch.tensor([total / (num_classes * counts[c]) for c in classes], dtype=torch.float32)

    sample_weights = [class_weights[y].item() for _, y in train_ds.samples]
    train_sampler = None
    if cfg.balance == "sampler":
        train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    loss_weights = class_weights if cfg.balance == "weights" else None

    meta = {
        "class_to_idx": train_ds.class_to_idx,
        "idx_to_class": {int(v): k for k, v in train_ds.class_to_idx.items()},
        "train_counts": counts,
    }
    return train_loader, val_loader, test_loader, loss_weights, meta


def build_model(cfg: TrainConfig, num_classes: int = 2) -> nn.Module:
    if cfg.model == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if cfg.use_pretrained else None
        model = torchvision.models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif cfg.model == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if cfg.use_pretrained else None
        model = torchvision.models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {cfg.model}")
    return model


def plot_curves(history: list, out_path: Path):
    if not history:
        return
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    train_acc = [100.0 * h['train_acc'] for h in history]
    val_macro_f1 = [h['val']['macro_f1'] for h in history]

    plt.figure(figsize=(11, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss)
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.grid(True, alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_acc)
    plt.xlabel('epoch'); plt.ylabel('acc %'); plt.grid(True, alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_macro_f1)
    plt.xlabel('epoch'); plt.ylabel('macro-F1'); plt.ylim(0, 1); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_confusion(cm: np.ndarray, class_names: list, out_path: Path, title: str):
    plt.figure(figsize=(4.2, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True'); plt.xlabel('Pred'); plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_roc(y_true: np.ndarray, y_scores: np.ndarray, out_path: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(4.2, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(title); plt.legend(loc="lower right"); plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, return_probs: bool = False) -> Dict:
    model.eval()
    all_logits, all_targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)
    y_true = targets.numpy()

    p, r, f1, support = precision_recall_fscore_support(y_true, preds, average=None, zero_division=0)
    macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)

    roc_auc = None
    try:
        if probs.shape[1] == 2:
            roc_auc = roc_auc_score(y_true, probs[:, 1])
        else:
            roc_auc = roc_auc_score(y_true, probs, multi_class='ovr')
    except Exception:
        roc_auc = None

    cm = confusion_matrix(y_true, preds)
    report = classification_report(y_true, preds, zero_division=0, output_dict=True)

    result = {
        "per_class_precision": p.tolist(),
        "per_class_recall": r.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "report": report,
    }
    if return_probs:
        result["y_true"] = y_true.tolist()
        result["probs"] = probs.tolist()
    return result


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    train_loader, val_loader, test_loader, loss_weights, meta = make_dataloaders(cfg)

    num_classes = len(train_loader.dataset.classes)
    model = build_model(cfg, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device) if loss_weights is not None else None)

    # Freeze backbone parameters initially
    backbone_params = []
    head_params = []
    if isinstance(model, torchvision.models.ResNet):
        backbone = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4)
        head = [model.fc]
        backbone_params = list(backbone.parameters())
        head_params = list(model.fc.parameters())
    else:
        # EfficientNet or others: assume last layer is classifier[-1]
        for name, p in model.named_parameters():
            if name.startswith('classifier.'):
                head_params.append(p)
            else:
                backbone_params.append(p)

    def set_backbone_trainable(trainable: bool):
        for p in backbone_params:
            p.requires_grad = trainable
        for p in head_params:
            p.requires_grad = True

    set_backbone_trainable(False)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = math.ceil(len(train_loader) * cfg.epochs)
    warmup_steps = max(1, int(cfg.warmup_epochs * len(train_loader)))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    use_amp = cfg.mixed_precision and device.type in ('cuda', 'mps')
    amp_dtype = torch.float16 if device.type in ('cuda', 'mps') else None
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == 'cuda'))

    out_dir = Path(f"{cfg.output_dir}_e{cfg.epochs}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(out_dir / 'dataset_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    best_metric = -1.0
    best_epoch = 0
    epochs_no_improve = 0

    global_step = 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", ncols=100)
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else torch.autocast(enabled=False, device_type='cpu')
            with ctx:
                outputs = model(images)
                loss = criterion(outputs, targets)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            global_step += 1

            pbar.set_postfix({
                'loss': f"{epoch_loss/total:.4f}",
                'acc': f"{100.0*correct/total:.2f}%",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        # Unfreeze after freeze_epochs
        if epoch == cfg.freeze_epochs:
            set_backbone_trainable(True)
            optimizer = optim.AdamW(model.parameters(), lr=cfg.lr * 0.3, weight_decay=cfg.weight_decay)
            scheduler = LambdaLR(optimizer, lr_lambda)

        # Validation
        val_metrics = evaluate(model, val_loader, device, return_probs=True)
        macro_f1 = val_metrics["macro_f1"]

        # Save checkpoint
        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'val_metrics': val_metrics,
            'config': asdict(cfg),
            'meta': meta,
        }
        _save_checkpoint(state, is_best=False, out_dir=out_dir, tag=f"epoch{epoch:03d}", keep_last_k=cfg.save_last_k)

        improved = macro_f1 > best_metric
        if improved:
            best_metric = macro_f1
            best_epoch = epoch
            epochs_no_improve = 0
            _save_checkpoint(state, is_best=True, out_dir=out_dir, tag=f"epoch{epoch:03d}", keep_last_k=cfg.save_last_k)
        else:
            epochs_no_improve += 1

        record = {
            'epoch': epoch,
            'train_loss': epoch_loss / max(1, total),
            'train_acc': correct / max(1, total),
            'val': val_metrics,
            'best_metric': best_metric,
            'best_epoch': best_epoch,
        }
        history.append(record)
        with open(out_dir / 'history.jsonl', 'a') as f:
            f.write(json.dumps(record) + "\n")

        # Plots
        try:
            plot_curves(history, out_dir / 'curves.png')
            class_names = sorted(train_loader.dataset.class_to_idx, key=train_loader.dataset.class_to_idx.get)
            cm = np.array(val_metrics['confusion_matrix'])
            plot_confusion(cm, class_names, out_dir / f'val_cm_epoch{epoch:03d}.png', title=f'Val Confusion (epoch {epoch})')
            if val_metrics.get('probs') is not None and len(class_names) == 2:
                y_true = np.array(val_metrics['y_true'])
                probs = np.array(val_metrics['probs'])
                plot_roc(y_true, probs[:, 1], out_path=out_dir / f'val_roc_epoch{epoch:03d}.png', title=f'Val ROC (epoch {epoch})')
        except Exception as e:
            print(f"Plotting failed: {e}")

        if epochs_no_improve >= cfg.early_stop_patience:
            print("Early stopping: no improvement.")
            break

    # Test on best
    best_ckpt = out_dir / 'best.pt'
    if test_loader is not None and best_ckpt.exists():
        print("Evaluating best checkpoint on TEST…")
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state['model_state'])
        test_metrics = evaluate(model, test_loader, device, return_probs=True)
        with open(out_dir / 'test_metrics.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print("Test macro-F1:", f"{test_metrics['macro_f1']:.4f}")
        try:
            class_names = sorted(train_loader.dataset.class_to_idx, key=train_loader.dataset.class_to_idx.get)
            cm = np.array(test_metrics['confusion_matrix'])
            plot_confusion(cm, class_names, out_dir / 'test_cm.png', title='Test Confusion')
            if test_metrics.get('probs') is not None and len(class_names) == 2:
                y_true = np.array(test_metrics['y_true'])
                probs = np.array(test_metrics['probs'])
                plot_roc(y_true, probs[:, 1], out_path=out_dir / 'test_roc.png', title='Test ROC')
        except Exception as e:
            print(f"Test plotting failed: {e}")


def _save_checkpoint(state: dict, is_best: bool, out_dir: Path, tag: str, keep_last_k: int = 3):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"checkpoint_{tag}.pt"
    torch.save(state, ckpt_path)
    ckpts = sorted(out_dir.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in ckpts[keep_last_k:]:
        try:
            old.unlink()
        except Exception:
            pass
    if is_best:
        shutil.copy2(ckpt_path, out_dir / 'best.pt')


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train binary damaged/undamaged classifier")
    p.add_argument('--data-dir', type=str, default='severity-classifier/binary_dataset')
    p.add_argument('--output-dir', type=str, default='severity-classifier/outputs/resnet50_384')
    p.add_argument('--img-size', type=int, default=384)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--warmup-epochs', type=int, default=1)
    p.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'efficientnet_b0'])
    p.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights')
    p.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    p.set_defaults(pretrained=True)
    p.add_argument('--freeze-epochs', type=int, default=3)
    p.add_argument('--early-stop-patience', type=int, default=5)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--no-amp', dest='amp', action='store_false')
    p.set_defaults(amp=True)
    p.add_argument('--balance', type=str, default='sampler', choices=['sampler', 'weights', 'none'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-last-k', type=int, default=3)
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])

    args = p.parse_args()
    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        model=args.model,
        use_pretrained=args.pretrained,
        freeze_epochs=args.freeze_epochs,
        early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
        mixed_precision=args.amp,
        balance=args.balance,
        seed=args.seed,
        save_last_k=args.save_last_k,
        device=args.device,
    )


def main():
    cfg = parse_args()
    train(cfg)


if __name__ == '__main__':
    main()
