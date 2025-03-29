from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

import tempfile
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import pydicom
import cv2
import matplotlib.pyplot as plt

router = APIRouter()

class MSClassifier(nn.Module):
    def __init__(self):
        super(MSClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=False)  
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)

# === IMAGE LOADER ===
def load_image(path):
    if path.lower().endswith(('.dcm', '.dicom')):
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array
        img = cv2.resize(img, (224, 224))
        img = img / np.max(img)
        return img
    elif path.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        return img
    else:
        raise ValueError("Unsupported file format.")

@router.post("/")
async def predict_ms(file : UploadFile = File(...)):

    try:
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".dcm", ".dicom")):
            raise HTTPException(status_code=400, detail="File must be a JPEG, PNG or DICOM Image")
        
        image_data = await file.read()

        # open and process the image here
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name



        img = load_image(tmp_path)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        img_tensor = normalize(img_tensor)
        
        #Use the model to predict the disease 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MSClassifier().to(device)

        model_path = 'C:/Users/sehza/Desktop/TestFolder/backend/routers/best_ms_classifier1.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            output = model(img_tensor.to(device))
            probability = output.item()
        
        # result here 
        diagnosis = "Multiple Sclerosis" if probability > 0.5 else "Healthy"
        confidence = probability if probability > 0.5 else 1 - probability

        result = {
            "diagnosis": diagnosis,
            "confidence": round(confidence * 100, 2),
            "ms_probability": round(probability, 4)
        }

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
