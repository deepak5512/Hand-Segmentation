import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from tqdm import tqdm

from src.dataset import SegDataset
from src.metrics import iou_score, pixel_accuracy, confusion_counts


def build_model(device):
    """
    DeepLabV3-ResNet50 (2 classes: background, hand).
    """
    weights = models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = models.segmentation.deeplabv3_resnet50(weights=weights)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    return model.to(device)


def _eval_split(dataloader, model, criterion, device):
    """Evaluate on one split (val/test)."""
    model.eval()
    tot_loss = tot_iou = tot_acc = 0.0
    tp = fp = fn = 0.0
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.squeeze(1).long().to(device)
            outputs = model(imgs)["out"]
            loss = criterion(outputs, masks)

            tot_loss += loss.item()
            tot_iou += iou_score(outputs, masks)
            tot_acc += pixel_accuracy(outputs, masks)
            ttp, tfp, tfn, _ = confusion_counts(outputs, masks)
            tp += ttp
            fp += tfp
            fn += tfn

    n = len(dataloader)
    if n == 0:
        return (0,)*6
    loss = tot_loss/n
    iou = tot_iou/n
    acc = tot_acc/n
    prec = tp/(tp+fp+1e-8)
    rec = tp/(tp+fn+1e-8)
    f1 = 2*prec*rec/(prec+rec+1e-8)
    return loss, iou, acc, prec, rec, f1


def train_val_test(
    data_root="data/egohands/processed_flat",
    epochs=10,
    batch_size=4,
    lr=1e-4,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    device=None,
):
    """
    Train/Val/Test pipeline (70/15/15 default).
    Saves everything under segmentation/newcheckpoints/
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.join("segmentation", "newcheckpoints")
    os.makedirs(base_dir, exist_ok=True)

    dataset = SegDataset(data_root, "images", "masks")
    n = len(dataset)
    n_train = int(n*train_ratio)
    n_val = int(n*val_ratio)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Split sizes → train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_iou, best_path = 0.0, os.path.join(base_dir, "best_model.pth")

    for epoch in range(1, epochs+1):
        model.train()
        tl, ti, ta, ttp, tfp, tfn = 0, 0, 0, 0, 0, 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            imgs = imgs.to(device)
            masks = masks.squeeze(1).long().to(device)
            opt.zero_grad()
            out = model(imgs)["out"]
            loss = criterion(out, masks)
            loss.backward()
            opt.step()
            tl += loss.item()
            with torch.no_grad():
                ti += iou_score(out, masks)
                ta += pixel_accuracy(out, masks)
                p, q, r, _ = confusion_counts(out, masks)
                ttp += p; tfp += q; tfn += r
        n = len(train_loader)
        tl/=n; ti/=n; ta/=n
        tp = ttp; fp = tfp; fn = tfn
        pr = tp/(tp+fp+1e-8); rc = tp/(tp+fn+1e-8)
        f1 = 2*pr*rc/(pr+rc+1e-8)

        vl, vi, va, vpr, vrc, vf1 = _eval_split(val_loader, model, criterion, device)
        print(f"[Train] {epoch}: L={tl:.4f} IOU={ti:.4f} Acc={ta:.4f} P={pr:.4f} R={rc:.4f} F1={f1:.4f}")
        print(f"[Val]   {epoch}: L={vl:.4f} IOU={vi:.4f} Acc={va:.4f} P={vpr:.4f} R={vrc:.4f} F1={vf1:.4f}")

        torch.save(model.state_dict(), os.path.join(base_dir, f"model_epoch_{epoch}_valsplit.pth"))
        if vi>best_iou:
            best_iou=vi
            torch.save(model.state_dict(), best_path)
            print(f"--> New best model saved (Val IOU={best_iou:.4f})")

    print("\nEvaluating best model on TEST set...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    tl, ti, ta, pr, rc, f1 = _eval_split(test_loader, model, criterion, device)
    print(f"[Test]  L={tl:.4f} IOU={ti:.4f} Acc={ta:.4f} P={pr:.4f} R={rc:.4f} F1={f1:.4f}")
    return model


def train_model_with_val(*a, **kw):
    """Alias for compatibility."""
    return train_val_test(*a, **kw)
