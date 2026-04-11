import cv2
import numpy as np
import os

def sharpness_score(image_path: str) -> float:
    """
    Score image sharpness using Laplacian variance.
    Higher = sharper. Typical range: 0 to 1000+
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    score = cv2.Laplacian(img, cv2.CV_64F).var()
    return round(float(score), 2)

def score_folder(folder_path):
        list_rate = []
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg'):
             jpgimages = os.path.join(folder_path, filename) 
             result = sharpness_score(jpgimages)
             list_rate.append({"filename": filename, "score": result})
        return list_rate