import cv2
import torch
from src.infer import load_model, segment_frame


def realtime_segmentation(
    model_path="checkpoints/model_epoch_10.pth",
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from: {model_path} on {device}")
    model = load_model(model_path, device=device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = segment_frame(model, frame, device=device)  # 0=bg, 1=hand

        # Simple overlay: color hand region
        overlay = frame.copy()
        overlay[mask == 1] = (0, 255, 0)  # green for hand

        alpha = 0.5
        vis = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.imshow("Hand Segmentation", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
