from src.train import train_model                 # old: full-data training
from src.train1 import train_val_test             # new: train/val/test version
from src.realtime import realtime_segmentation


def main():
    print("1. Train on full data (no val/test)")
    print("2. Train + Val + Test (70/15/15)  → saves to segmentation/newcheckpoints/")
    print("3. Run real-time hand segmentation (uses best_model.pth)")
    choice = input("Enter choice: ").strip()

    if choice == "1":
        train_model(epochs=10)
    elif choice == "2":
        train_val_test(epochs=10)
    elif choice == "3":
        model_path = "segmentation/newcheckpoints/best_model.pth"
        realtime_segmentation(model_path=model_path)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
