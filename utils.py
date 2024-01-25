import matplotlib.pyplot as plt
import os
from datetime import datetime


direction_vectors = {
    "N":  (0, 1),
    "NE": (1, 1),
    "E":  (1, 0),
    "SE": (1, -1),
    "S":  (0, -1),
    "SW": (-1, -1),
    "W":  (-1, 0),
    "NW": (-1, 1)
} # 0维左右，1维上下


def draw_loss(epoch, train_losses, valid_losses):
    plt.figure()
    epochs = range(1, epoch + 1)
    plt.plot(epochs, train_losses, label="Training loss")
    plt.plot(epochs, valid_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.now().strftime("%m_%d_%H_%M")
    filename = f"loss_{current_time}.png"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath)
    plt.close()

    print(f"Save pic: loss_{current_time}.png")

