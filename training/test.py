import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 64)  # Adjust based on image size
        self.fc2 = nn.Linear(64, 2)  # 2 classes: MS or Healthy

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1) 
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNNModel()  
model.load_state_dict(torch.load("best_ms_classifier.pth"))
model.to(device)  
model.eval()  

def test_model_on_directory(test_directory):  
    correct_predictions = 0
    total_images = 0

    for file in os.listdir(test_directory):
        label = 0 if "healthy" in file else 1  
        if not file.endswith('jpg'):
            continue
        file_path = os.path.join(test_directory, file)

        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if img is None:
            print(f"Error reading image {file_path}. Skipping.")
            continue  
        img = cv2.resize(img, (224, 224))  # Resize to match training size
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = np.expand_dims(img, axis=1)  # Add channel dimension (for grayscale: 1 channel)
        img = img.astype(np.float32)  # Convert to float32
        img = torch.tensor(img).to(device)  # Convert to tensor and move to device

        with torch.no_grad():
            output = model(img)
            # print(f"Output: {output} of {file}")
            probabilities = F.softmax(output, dim=1) 
            
            _, prediction = torch.max(output, 1)  # Get the class with the highest probability
            print(f"Probabilities: {probabilities} of {file} with label {label}, prediction {prediction.item()}")


        if prediction.item() == label:
            correct_predictions += 1
        total_images += 1

    # Print accuracy
    accuracy = (correct_predictions / total_images) * 100
    print(f"Accuracy on {test_directory} dataset: {accuracy:.2f}%")

test_model_on_directory("combination") 

