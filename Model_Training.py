import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# images directory path should be ./dataset/new/dataset/and 4 folders (Back, Forward, Left, Right).
base_directory = './dataset/new/dataset'  # Path to the folder containing 'Left', 'Right', 'Forward', 'Down'
class_folders = ['Left', 'Right', 'Forward', 'Back']  # Folder names correspond to class labels
class_labels = {folder: idx for idx, folder in enumerate(class_folders)}  # Assign labels: 0=Left, 1=Right, etc.

# Preprocess and load all images
images = []
labels = []

for folder in class_folders:
    folder_path = os.path.join(base_directory, folder)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image files
            # Read the image, preprocess (grayscale, resize, normalize), and append
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (128, 128))  # Resize to match model input size
            img_normalized = img_resized / 255.0     # Normalize pixel values to [0, 1]
            images.append(img_normalized.flatten())  # Flatten the image to 1D vector
            labels.append(class_labels[folder])      # Append the corresponding label

# Convert to PyTorch tensors
X_train = torch.tensor(images, dtype=torch.float32)
y_train = torch.tensor(labels, dtype=torch.long)


# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(128 * 128, 512) # Input layer with 128x128 pixels; Hidder layer 1
        self.fc2 = nn.Linear(512, 128) # Hidden layer 2
        self.fc3 = nn.Linear(128,64) # Hidder layer 3
        self.fc4 = nn.Linear(64, 4) # Output layer
    
    # Relu activation on Input layer, Hidder layer 1 and 2.
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)        
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.000001)  # Adam optimizer with learning rate

# Train the model
epochs = 450  # Number of epochs for training
for epoch in range(epochs):
    running_loss = 0.0
    for i in range(len(X_train)):
        optimizer.zero_grad()  # Clear gradients
        output = model(X_train[i].view(1, -1))  # Forward pass (1 image at a time)
        loss = criterion(output, y_train[i].view(1))  # Compute loss
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update weights
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(X_train):.4f}")

torch.save(model.state_dict(), "model3.pth")
print("\nModel saved\n")
# Evaluate the model on the test set
correct = 0
total = len(X_train)
with torch.no_grad():  # Disable gradient tracking for testing
    for i in range(total):
        output = model(X_train[i].view(1, -1))
        _, predicted = torch.max(output, 1)
        correct += (predicted == y_train[i]).sum().item()
accuracy = 100 * correct / total
print(f"Accuracy on the test set: {accuracy:.2f}%")

# Visualize the test set predictions
new_image = "./dataset/new/dataset/image/new_image.jpg"
new_img = cv2.imread(new_image, cv2.IMREAD_GRAYSCALE)
new_img_resized = cv2.resize(new_img, (128, 128))       # Resize to match the training data
new_img_normalized = new_img_resized / 255.0          # Normalize pixel values
new_img_flattened = new_img_normalized.flatten()      # Flatten into 1D vector
new_img_tensor = torch.tensor(new_img_flattened, dtype=torch.float32)

with torch.no_grad():
    output = model(new_img_tensor.view(1, -1))
    _, predicted = torch.max(output.data, 1)
print(f"Predicted label: {predicted.item()}, Actual label: {y_train[i].item()}")
plt.imshow(new_img_tensor.view(128, 128).numpy(), cmap='gray')
plt.title(f"True: {y_train[i].item()}, Pred: {predicted.item()}")
plt.show()
