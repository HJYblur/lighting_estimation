import cv2
import numpy as np


# 阈值设定为127（在0到255的灰度值范围内）
def check_intensity(x, threshold=127):
    points = [(0, 100000), (127, 200000), (256, 400000)]
    
    if x <= threshold:
        (x1, y1), (x2, y2) = points[0], points[1]
    else:
        (x1, y1), (x2, y2) = points[1], points[2]
        
    y = y1 + (y2 - y1) / (x2 - x1) * (x - x1)
    return y

def analyze_brightness(path):
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness_values = gray_img.flatten()

    top_8_percent = np.percentile(brightness_values, 92)
    top_mean = np.mean(top_8_percent) if top_8_percent.size > 0 else 0

    intensity = check_intensity(top_mean)
    
    remaining_pixels = brightness_values[brightness_values < top_8_percent]
    remaining_mean = np.mean(remaining_pixels) / 256 if remaining_pixels.size > 0 else 0
    
    return intensity, remaining_mean
    
    