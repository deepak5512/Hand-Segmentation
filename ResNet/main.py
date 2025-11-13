from src.train import train_model
from src.realtime import realtime_segmentation

if __name__ == "__main__":
    print("1. Train model")
    print("2. Run real-time hand segmentation")
    choice = int(input("Enter choice: "))

    if choice == 1:
        train_model(epochs=10)
    elif choice == 2:
        realtime_segmentation()
