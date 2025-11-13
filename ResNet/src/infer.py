import torch
from torchvision import models, transforms
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
import cv2
import numpy as np


def load_model(path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # SAME setup as in training
    weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = models.segmentation.deeplabv3_resnet50(weights=weights)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=1)

    # Load your trained weights (ignore aux mismatches if any)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    return model


def segment_frame(model, frame, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    h, w = frame.shape[:2]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)["out"]
        mask = torch.argmax(out, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # Resize mask back to original frame size
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask
