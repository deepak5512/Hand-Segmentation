import os
import shutil
from tqdm import tqdm

src_img_root = "../data/egohands/processed/images"
src_mask_root = "../data/egohands/processed/masks"
dst_img = "../data/egohands/processed_flat/images"
dst_mask = "../data/egohands/processed_flat/masks"

os.makedirs(dst_img, exist_ok=True)
os.makedirs(dst_mask, exist_ok=True)

counter = 0
for subfolder in tqdm(os.listdir(src_img_root), desc="Flattening EgoHands"):
    img_dir = os.path.join(src_img_root, subfolder)
    mask_dir = os.path.join(src_mask_root, subfolder)

    # ✅ Skip if not a folder
    if not os.path.isdir(img_dir):
        continue
    
    # ✅ Skip if corresponding mask folder is missing
    if not os.path.isdir(mask_dir):
        print(f"Warning: Missing mask directory for {subfolder}, skipping.")
        continue

    # Loop through the IMAGE files
    for img_file in os.listdir(img_dir):
        # Get the image name and extension
        img_name, img_ext = os.path.splitext(img_file)

        # Skip non-jpg files in the image folder
        if not img_ext.lower() in [".jpg", ".jpeg"]:
            continue

        # --- KEY FIX ---
        # 1. Construct the *expected* mask file name (e.g., "frame_0001.png")
        mask_file = img_name + ".png"
        
        # 2. Construct the full paths for both
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        # --- END OF FIX ---

        # ✅ Skip if corresponding .png mask is missing
        if not os.path.exists(mask_path):
            print(f"Warning: Missing mask {mask_path} for image {img_path}, skipping.")
            continue

        # --- KEY FIX 2 ---
        # Generate new names, preserving the different extensions
        new_img_name = f"egohands_{counter:06d}{img_ext}"  # e.g., egohands_000000.jpg
        new_mask_name = f"egohands_{counter:06d}.png"      # e.g., egohands_000000.png
        # --- END OF FIX ---

        shutil.copy(img_path, os.path.join(dst_img, new_img_name))
        shutil.copy(mask_path, os.path.join(dst_mask, new_mask_name))
        counter += 1

print(f"✅ Flattened {counter} image-mask pairs into {dst_img} and {dst_mask}")