import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pydicom
import cv2
from sklearn.metrics import roc_auc_score
from pydicom import dcmread
    

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
            ms_images.append(img)  

healthy_folder = "healthy/ST000001"    

for folder in os.listdir(healthy_folder):
    for file in os.listdir(f"{healthy_folder}/{folder}"):
        if file.endswith(".dcm"):
            path = os.path.join(healthy_folder, folder, file)
            img = load_dicom(path)
            healthy_images.append(img)  

class MRIDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PyTorch tensor
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dim
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

# Convert numpy arrays
ms_images = np.array(ms_images)
healthy_images = np.array(healthy_images)

# Create labels
ms_labels = np.ones(len(ms_images))
healthy_labels = np.zeros(len(healthy_images))

# Combine data
X = np.concatenate([ms_images, healthy_images], axis=0)
y = np.concatenate([ms_labels, healthy_labels], axis=0)

# Create dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure correct size
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

dataset = MRIDataset(X, y, transform=transform)

# Split into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size], 
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class MSClassifier(nn.Module):
    def __init__(self):
        super(MSClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # Use smaller backbone
        
        # Modify first layer to accept 1 channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final layer with stronger regularization
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),  # Add batch normalization
            nn.Dropout(0.6),      # Increase dropout
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
            
    def forward(self, x):
        return self.resnet(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MSClassifier().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)


# Training function
def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=10):
    best_val_auc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            pred = (outputs > 0.5).float()
            correct += (pred == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                
                val_loss += loss.item() * inputs.size(0)
                pred = (outputs > 0.5).float()
                val_correct += (pred == labels.unsqueeze(1)).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_val_loss = val_loss / len(test_loader.dataset)
        epoch_val_acc = val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # Calculate AUC
        val_auc = roc_auc_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, "
              f"Val AUC: {val_auc:.4f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_ms_classifier1.pth')
            print(f"Model saved with AUC: {val_auc:.4f}")
    
    model.load_state_dict(torch.load('best_ms_classifier1.pth'))
    return model, (train_losses, val_losses, train_accs, val_accs)

# Train model
model, history = train_model(model, criterion, optimizer, train_loader, test_loader)

# Evaluate final model
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend((outputs > 0.5).float().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'MS'],
                yticklabels=['Healthy', 'MS'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

# Plot training history
def plot_history(history):
    train_losses, val_losses, train_accs, val_accs = history
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs)
    plt.plot(val_accs)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Evaluate and plot history
evaluate_model(model, test_loader)
# plot_history(history)

# Function to predict on new DICOM images
def predict_ms(model, image_path):

    dicom = dcmread(image_path)
    img = dicom.pixel_array
    img = cv2.resize(img, (224, 224))
    img = img / np.max(img)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    
    # Apply normalization
    img_tensor = transforms.Normalize(mean=[0.5], std=[0.5])(img_tensor)
    
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.to(device))
        probability = output.item()
    
    diagnosis = "MS" if probability > 0.5 else "Healthy"
    print(f"Prediction for {image_path}:")
    print(f"Probability of MS: {probability:.4f}")
    print(f"Diagnosis: {diagnosis}")
    
    return probability

# Save the prediction function for future use
print("Model saved as 'best_ms_classifier.pth'")
print("Use the predict_ms function to make predictions on new images")