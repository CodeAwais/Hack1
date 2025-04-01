import os
import pydicom
import numpy as np
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torchvision import transforms
import torch.nn.functional as F
from sklearn.metrics import classification_report


def load_dicom(path):
    dicom = pydicom.dcmread(path)
    img = dicom.pixel_array
    img = cv2.resize(img, (224, 224))
    img = img / np.max(img)
    return img

ms_images = []
healthy_images = []

ms_folder = "multiple_sclerosis/ST000001"

for folder in os.listdir(ms_folder):
    for file in os.listdir(f"{ms_folder}/{folder}"):
        if file.endswith(".dcm"):
            path = os.path.join(ms_folder, folder, file)
            img = load_dicom(path)
            ms_images.append(img)  # Append image to list

healthy_folder = "healthy/ST000001"    

for folder in os.listdir(healthy_folder):
    for file in os.listdir(f"{healthy_folder}/{folder}"):
        if file.endswith(".dcm"):
            path = os.path.join(healthy_folder, folder, file)
            img = load_dicom(path)
            healthy_images.append(img)    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

labels = [1] * len(ms_images) + [0] * len(healthy_images)  # 1 for MS, 0 for healthy 

X = np.array(ms_images + healthy_images)
y = np.array([1] * len(ms_images) + [0] * len(healthy_images))  

print("Dataset shape:", X.shape)  
print("Labels shape:", y.shape) 

X = np.expand_dims(X, axis=1)  # (656, 1, 224, 224)
X = X.astype(np.float32)  # Convert to float32 for deep learning

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train shape:", X_train.shape)  # (524, 1, 224, 224)
print("Test shape:", X_test.shape)    # (132, 1, 224, 224)


class MRIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # Convert to tensor
        self.y = torch.tensor(y, dtype=torch.long)     # Convert labels to tensor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create datasets and dataloaders
train_dataset = MRIDataset(X_train, y_train)
test_dataset = MRIDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


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
        x = x.view(x.shape[0], -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

# After evaluating the model on the test set
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())  # Move predictions to CPU
        true_labels.extend(labels.cpu().numpy())  # Move labels to CPU

accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Get classification report
report = classification_report(true_labels, predictions, target_names=['Healthy', 'MS'])
print("Classification Report:\n", report)

# Save model as before
torch.save(model.state_dict(), "mri_model.pth")
print("Model saved to mri_model.pth")

