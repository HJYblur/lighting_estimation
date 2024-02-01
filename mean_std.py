import os
from skimage import io
import numpy as np
from tqdm import tqdm
from data_loader import TRAIN_PATH

## mean: [0.3422852  0.33595875 0.24629457] std: [0.27514981 0.25132029 0.23664711]

count = 0
total_mean = np.zeros(3)
M2 = np.zeros(3)

for img_name in tqdm(os.listdir(TRAIN_PATH), desc = 'Precessing images'):
    image = io.imread("../Data/train/" + img_name)[:, :, :3] / 255.0
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            count += 1
            val = image[x, y, :]
            delta = val - total_mean
            total_mean = total_mean + delta / count
            delta2 = val - total_mean
            M2 = M2 + delta * delta2

print("mean:", total_mean, "std:", np.sqrt(M2 / (count - 1)))
