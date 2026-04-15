import cv2
import numpy as np
import os
from ultralytics import YOLO

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
        list_rate.sort(key=lambda x: x['score'], reverse=True)
        return list_rate

def detect_people(image_path): 
    model = YOLO("yolov8n.pt")
    results = model(image_path)
    person_count = int (len(results[0].boxes.cls))
    biggest = max(results[0].boxes.xywh, key=lambda box: box[2] * box[3])
    subject_size = float(biggest[2] * biggest[3]) / (results[0].orig_shape[0] * results[0].orig_shape[1])
    return {"person_count": person_count, "subject_size": subject_size}

def standard_score(image_path):
    result1 = sharpness_score(image_path)
    normalized = min(result1 / 1000, 1.0)
    result2 = detect_people(image_path)
    mid_score = (normalized * 0.25) + result2['subject_size'] *0.25
    return mid_score
