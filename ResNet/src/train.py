import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from src.dataset import SegDataset
from src.metrics import iou_score, pixel_accuracy, confusion_counts
import os

def train_model(data_root='data/egohands/processed_flat', epochs=10, batch_size=4, lr=1e-4, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure checkpoint directory exists
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Dataset + loader
    dataset = SegDataset(
        parent_dir=data_root,
        image_dir="images",
        mask_dir="masks"
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Model: DeepLabV3-ResNet50 with 2 classes
    weights = models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = models.segmentation.deeplabv3_resnet50(weights=weights)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        iou_total = 0.0
        acc_total = 0.0
        tp_total = 0.0
        fp_total = 0.0
        fn_total = 0.0

        for imgs, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)

            # masks: [N, 1, H, W] float -> [N, H, W] long (class indices 0/1)
            masks = masks.squeeze(1).long().to(device)

            optimizer.zero_grad()
            outputs = model(imgs)["out"]  # [N, 2, H, W]

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Detach for metrics
            with torch.no_grad():
                iou_total += iou_score(outputs, masks)
                acc_total += pixel_accuracy(outputs, masks)
                tp, fp, fn, tn = confusion_counts(outputs, masks)
                tp_total += tp
                fp_total += fp
                fn_total += fn

        num_batches = len(dataloader)
        avg_loss = running_loss / num_batches
        avg_iou = iou_total / num_batches
        avg_acc = acc_total / num_batches

        precision = tp_total / (tp_total + fp_total + 1e-8)
        recall    = tp_total / (tp_total + fn_total + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        print(
            f"Loss: {avg_loss:.4f}, "
            f"IOU: {avg_iou:.4f}, "
            f"Acc: {avg_acc:.4f}, "
            f"Prec: {precision:.4f}, "
            f"Rec: {recall:.4f}, "
            f"F1: {f1:.4f}"
        )

        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pth"))

    return model
