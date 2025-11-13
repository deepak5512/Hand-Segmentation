import scipy.io
import numpy as np
import cv2
import os

# --- Configuration ---
root_dir = "../data/egohands/egohands_data/_LABELLED_SAMPLES"
output_dir = "../data/egohands/processed"
# ---------------------

print(f"Starting processing from: {root_dir}")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

# Iterate over each video folder (e.g., "CAR_1_A")
for folder in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    mat_path = os.path.join(folder_path, "polygons.mat")
    if not os.path.exists(mat_path):
        print(f"Warning: 'polygons.mat' not found in {folder}. Skipping.")
        continue

    print(f"Processing: {folder}")

    # Load MATLAB file
    # struct_as_record=False is key to get object-like structs
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    polygons = mat["polygons"]

    # --- FIX 1: Normalize polygons array ---
    # The 'polygons' object can be (100,) or (100, 1) or (1, 100).
    # .flatten() ensures it's always a 1D array of (100,)
    # so we can access each frame's struct by index.
    if polygons.ndim > 0:
        frames_polygons = polygons.flatten()
    else:
        # Handle edge case if it's a single squeezed item
        frames_polygons = np.array([polygons.item()])

    frames = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
    out_img_dir = os.path.join(output_dir, "images", folder)
    out_mask_dir = os.path.join(output_dir, "masks", folder)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # Iterate over all frames in the folder
    for idx, frame_name in enumerate(frames):
        # Stop if mat file has fewer entries than frames
        if idx >= len(frames_polygons):
            print(f"Warning: More frames than polygon entries in {folder}. Stopping at frame {idx}.")
            break

        img_path = os.path.join(folder_path, frame_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
        
        # Create a new blank mask for this frame
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Get the struct object for this specific frame
        frame_data = frames_polygons[idx]

        # --- FIX 2: Iterate by attributes, not over the struct ---
        # The struct has 4 fields: myleft, myright, yourleft, yourright.
        # We must check each one.
        polys_to_iterate = [
            frame_data.myleft,
            frame_data.myright,
            frame_data.yourleft,
            frame_data.yourright
        ]

        for hand_poly in polys_to_iterate:
            # Check if the polygon data exists.
            # As seen in your image, '[]' is loaded as an empty array.
            # We check if it's a valid numpy array with data.
            if not isinstance(hand_poly, np.ndarray) or hand_poly.size == 0:
                continue

            # Convert to int32 for cv2.fillPoly
            pts = np.array(hand_poly, dtype=np.int32)

            # Check if it's a valid (N, 2) polygon with at least 3 points
            if pts.ndim == 2 and pts.shape[1] == 2 and len(pts) > 2:
                cv2.fillPoly(mask, [pts], 255)

        # Save the original image and the generated mask
        cv2.imwrite(os.path.join(out_img_dir, frame_name), img)
        # Save mask with a .png extension to avoid compression issues
        mask_name = os.path.splitext(frame_name)[0] + ".png"
        cv2.imwrite(os.path.join(out_mask_dir, mask_name), mask)

print(f"✅ All masks successfully generated at {os.path.join(output_dir, 'masks')}")