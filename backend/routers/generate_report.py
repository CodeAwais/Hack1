from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

import tempfile
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import pydicom
import cv2

load_dotenv()  # load the environment variables

router = APIRouter()

class DetailsRequest(BaseModel):
    age: int
    gender: str
    symptoms: list[str] = []
    family_history: bool
    smoking_history: bool
    epstein_barr_virus: bool


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
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.resnet(x)


# === IMAGE LOADER ===
def load_image(path):
    if path.lower().endswith((".dcm", ".dicom")):
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array
        img = cv2.resize(img, (224, 224))
        img = img / np.max(img)
        return img
    elif path.lower().endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        return img
    else:
        raise ValueError("Unsupported file format.")


@router.post("/")
async def generate_report(
    file: UploadFile = File(...),
    age: int = Form(...),
    gender: str = Form(...),
    symptoms: str = Form(""),
    family_history: bool = Form(...),
    smoking_history: bool = Form(...),
    epstein_barr_virus: bool = Form(...),
):
    try:
        # === MS Prediction ===
        if not file.filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".dcm", ".dicom")
        ):
            raise HTTPException(
                status_code=400, detail="File must be a JPEG, PNG or DICOM Image"
            )

        image_data = await file.read()

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename)[1]
        ) as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name

        img = load_image(tmp_path)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        img_tensor = normalize(img_tensor)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MSClassifier().to(device)

        # NOTE:  Replace with your actual model path
        model_path = "C:/Users/sehza/Desktop/TestFolder/backend/routers/best_ms_classifier1.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            output = model(img_tensor.to(device))
            probability = output.item()

        diagnosis = "Multiple Sclerosis" if probability > 0.5 else "Healthy"
        confidence = probability if probability > 0.5 else 1 - probability

        ms_result = {
            "diagnosis": diagnosis,
            "confidence": round(confidence * 100, 2),
            "ms_probability": round(probability, 4),
        }

        # === Report Generation ===
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500, detail="Perplexity API Key is not set or invalid"
            )

        symptoms_str = ""
        if symptoms:
            symptoms_str = (
                " The patient has the following symptoms " + symptoms + "."
            )

        prompt = (
            f"Generate a detailed report for the patient with the following details: "
            f"The age of the patient is {age} and the gender is {gender}."
            f"The MRI scan suggests {ms_result['diagnosis']} with {ms_result['confidence']}% for confidence."
            f"and the symptoms the patient reported are {symptoms_str}"
            f"The patient's family history is (true/false): {family_history} "
            f"The patient's smoking history is (true/false): {smoking_history}"
            f"The patitent has epstein barr virus (true/false): {epstein_barr_virus}"
            f"Add the following sections to the report: "
            f"1. Patient Summary"
            f"2. Risk Assessment"
            f"3. Key MRI features that support the diagnosis"
            f"Be medically accurate and use plain language."
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        body = {
            "model": "sonar",
            "messages": [{"role": "user", "content": prompt}],
        }

        response = requests.post(
            "https://api.perplexity.ai/chat/completions", headers=headers, json=body
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=500, detail=f"API request failed! {response.text}"
            )

        data = response.json()
        content = data.get("choices")[0].get("message").get("content")

        return {"summary": content, "ms_result": ms_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
