import torch

def pixel_accuracy(output, target):
    # output: [N, C, H, W], target: [N, H, W]
    pred = torch.argmax(output, dim=1)
    correct = (pred == target).float()
    return (correct.sum() / correct.numel()).item()

def iou_score(output, target, num_classes=2):
    # output: [N, C, H, W], target: [N, H, W]
    pred = torch.argmax(output, dim=1)
    ious = []

    for cls in range(num_classes):
        pred_c = (pred == cls)
        tgt_c = (target == cls)
        intersection = (pred_c & tgt_c).sum().float()
        union = (pred_c | tgt_c).sum().float()
        if union > 0:
            ious.append((intersection / union).item())

    if not ious:
        return 0.0
    return sum(ious) / len(ious)

def confusion_counts(output, target, positive_class=1):
    # output: [N, C, H, W], target: [N, H, W]
    pred = torch.argmax(output, dim=1)

    pos_pred = (pred == positive_class)
    pos_true = (target == positive_class)

    tp = (pos_pred & pos_true).sum().item()
    fp = (pos_pred & ~pos_true).sum().item()
    fn = (~pos_pred & pos_true).sum().item()
    tn = (~pos_pred & ~pos_true).sum().item()

    return tp, fp, fn, tn
