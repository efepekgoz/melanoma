import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import f1_score
import torchvision.transforms as transforms
import torch
from torch import nn, optim
from tqdm import tqdm


########################################
#   Prereqs.
#   test csv has no labels so I splitted train data 80/20 for testing
#   some data does not have jpeg version, only dicom and vice versa
#   so move jpegs into dicom's folder. Also adjust the paths for your own machine.
#   
#   Modified the original train.csv
#   Dataset split into 50/50 malignant beign ratio
#   which makes 584/584 in each label
#   With augmentations this number can be increased
#   
#   TODO:
#   used resnet50 got respectable .78 f1, can try different models
#   play with lr-bs, increase num epochs, add transforms, 
#   
########################################

#old dataset management
"""df = pd.read_csv('/Users/efepekgoz/Developer/EEEM068/Melanoma/mela/train.csv')
target_counts = df['target'].value_counts()
print("Number of target 0:", target_counts[0])
print("Number of target 1:", target_counts[1])

target_1_df = df[df['target'] == 1]
target_0_df = df[df['target'] == 0].head(584)
new_df = pd.concat([target_1_df, target_0_df])
new_df.to_csv('/Users/efepekgoz/Developer/EEEM068/Melanoma/mela/modified.csv', index=False)
modified_df = new_df.iloc[:, [0, -1]]

modified_df.to_csv('/Users/efepekgoz/Developer/EEEM068/Melanoma/mela/modified.csv', index=False)
"""


class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        base_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        img_path = self.find_file(base_path)

        file_extension = os.path.splitext(img_path)[-1].lower()

        """if file_extension in ['.dcm']:
            dicom = pydicom.dcmread(img_path)
            image = apply_voi_lut(dicom.pixel_array, dicom)
            if dicom.PhotometricInterpretation == "MONOCHROME1":
                image = np.amax(image) - image
            image = Image.fromarray(image).convert('RGB')"""
        if file_extension in ['.jpg', '.jpeg']:
            image = Image.open(img_path).convert('RGB')
        else:
            raise ValueError(f"Unsupported file extension: {file_extension} in file {img_path}")

        if self.transform:
            image = self.transform(image)

        label = self.annotations.iloc[idx, 1]
        return image, label
    def find_file(self, base_path):
        # Check if the base path already has a valid extension
        if os.path.isfile(base_path):
            return base_path
        # Try appending likely extensions and checking each
        for extension in ['.jpg', '.jpeg', '.dcm']:
            test_path = f"{base_path}{extension}"
            if os.path.isfile(test_path):
                return test_path
        raise FileNotFoundError(f"No file found for base path: {base_path}")   
    
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

print("Creating Dataset...")
full_dataset = ImageDataset(
    csv_file='/Users/efepekgoz/Developer/EEEM068/Melanoma/mela/modified.csv',
    root_dir='/Users/efepekgoz/Developer/EEEM068/Melanoma/mela/train/',
    transform=transform
)
print("Dataset complete!")
print("Splitting dataset...")
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
print("Split complete!")

print("assigning dataloaders...")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
print("dataloaders ready!")

print("loading model...")
model = resnet50(weights=ResNet50_Weights.DEFAULT)
print("resnet50 loaded!")

print("arranging features for classification...")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Binary classification
print("model modified for binary!")

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')   #switch mps for cpu if not macOS
model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("optimiser and loss fn ready!")

print("defining train and test funcs...")
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backprop n optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update the progress bar with the current loss
        progress_bar.set_description(f"Training Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def test_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    progress_bar = tqdm(test_loader, desc='Testing', leave=False)
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Update the progress bar with the current loss
            progress_bar.set_description(f"Test Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predictions, average='binary')  # 'binary' for binary classification

    return avg_loss, accuracy, f1
print("test and training created !")

num_epochs = 10
print("training begins...")
for epoch in range(num_epochs):
    train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')


test_loss, test_accuracy, test_f1 = test_model(model, test_loader, criterion, device)
print(f'Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Test F1 Score: {test_f1:.2f}')

#print("saving model...")
# torch.save(model.state_dict(), '/models/melanoma_classification_model.pth')


######
#
#  Original dataset 50/50
#  Epoch 10/10, Training Loss: 0.5459, Training Accuracy: 72.27%                                                                                        
#  Final Test Loss: 0.5336, Test Accuracy: 73.50%, Test F1 Score: 0.75 
#
#
