import cv2
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

import tempfile
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import pydicom
from dotenv import load_dotenv
import requests

load_dotenv()
router = APIRouter()


# === MSClassifier ===
class MSClassifier(nn.Module):
    def __init__(self):
        super(MSClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
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


# === Load DICOM Image ===
def load_dicom(path):
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array
    img = cv2.resize(img, (224, 224))
    img = img / np.max(img)
    return img


@router.post("/")
async def predict_and_generate_report(
    file: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    symptoms: str = Form(...),
    family_history: str = Form(False),
    smoking_history: str = Form(False),
    ebv: str = Form(False)
):
    try:
        # Check for DICOM
        if not file.filename.lower().endswith((".dcm", ".dicom")):
            raise HTTPException(status_code=400, detail="Only DICOM (.dcm) files are supported.")

        # Save temp file
        image_data = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name

        print(f"age: {age}, sex: {sex}, symptoms: {symptoms}, family_history: {family_history}, smoking_history: {smoking_history}, ebv: {ebv}")
        # Load and preprocess image
        img = load_dicom(tmp_path)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        img_tensor = normalize(img_tensor)

        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MSClassifier().to(device)
        model_path = os.path.join(os.path.dirname(__file__), 'best_ms_classifier1.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Predict
        with torch.no_grad():
            output = model(img_tensor.to(device))
            probability = output.item()

        diagnosis = "Likely Multiple Sclerosis" if probability > 0.5 else "Healthy"
        confidence = round(probability * 100 if probability > 0.5 else (1 - probability) * 100, 2)

        # 🔥 Generate prompt for Perplexity
        symptom_str = ", ".join(symptoms) if symptoms else "none"
        prompt = (
            f"Generate a concise medical report for a {age}-year-old {sex} patient. "
            f"The MRI scan analysis indicates: {diagnosis} with {confidence}% confidence. "
            f"Symptoms: {symptom_str}. "
            f"Family history of MS: {family_history}. "
            f"Smoking history: {smoking_history}. "
            f"Epstein-Barr virus history: {ebv}. "
            f"FORMAT: Provide three short paragraphs only: "
            f"1) Patient Summary (2-3 sentences about patient profile and symptoms) "
            f"2) Key MRI Features (2-3 sentences about imaging findings) "
            f"3) Recommendations (2-3 sentences with next steps) "
            f"IMPORTANT: Return plain text only. NO markdown formatting, NO bullet points, NO headers, NO asterisks. "
            f"Keep the entire response under 250 words. Be direct and concise."
        )

        # Call Perplexity API
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        headers = {
            "Authorization": f"Bearer {perplexity_api_key}",
            "Content-Type": "application/json"
        }

        body = {
            "model": "sonar",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=body)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Perplexity API error: {response.text}")

        summary = response.json()["choices"][0]["message"]["content"]

        return {
            "diagnosis": diagnosis,
            "confidence": confidence,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
