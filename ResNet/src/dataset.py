import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class SegDataset(Dataset):
    def __init__(self, parent_dir, image_dir, mask_dir, transform=None):
        self.image_path = os.path.join(parent_dir, image_dir)
        self.mask_path = os.path.join(parent_dir, mask_dir)
        self.images = sorted(os.listdir(self.image_path))
        self.masks = sorted(os.listdir(self.mask_path))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_path, self.images[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_path, self.masks[idx])).convert("L")

        transform_img = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        transform_mask = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

        img = transform_img(img)
        mask = transform_mask(mask)

        return img, mask
