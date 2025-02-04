import torch
import torch.nn as nn
import cv2
from picamera2 import Picamera2
import numpy as np
import paho.mqtt.client as mqtt
import time

broker = "broker.hivemq.com"
port = 1883
topic = "direction"
message = ""

client = mqtt.Client()

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(128 * 128, 512) 
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128,64)  
        self.fc4 = nn.Linear(64, 4) 
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)       
        return x


# Preprocess function for the image
def preprocess_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img_resized = cv2.resize(img_gray, (128, 128))       # Resize to 128x128
    img_normalized = img_resized / 255.0               # Normalize pixel values
    img_flattened = img_normalized.flatten()           # Flatten into 1D vector
    return torch.tensor(img_flattened, dtype=torch.float32)

# Load the trained model
model = SimpleNN()
model.load_state_dict(torch.load("model3.pth"))  # Load the trained model
model.eval()  # Set the model to evaluation mode

# Define class names
class_names = ["Left", "Right", "Forward", "Back"]

# Initialize the Pi Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))  # Configure resolution
picam2.start()

print("Press 'c' to capture an image and predict, or 'q' to quit.")

try:
    while True:
        # Capture live feed
        image = picam2.capture_array()
        cv2.imshow("Live Feed", image)  # Show the live camera feed

        # Detect key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Press 'c' to capture an image
            print("Capturing image...")
            
            # Call's Preprocess function to Preprocess the captured image
            image_tensor = preprocess_image(image)
            
            # Make a prediction
            with torch.no_grad():
                output = model(image_tensor.view(1, -1))  # Add batch dimension
                _, predicted = torch.max(output, 1)
            
            # Display the prediction
            predicted_label = predicted.item()
            print(f"Predicted Direction: {class_names[predicted_label]}")
                        # Publish the message
            message = predicted_label
            time.sleep(5)
            client.connect(broker, port)
            client.publish(topic, message)
            time.sleep(3)
            print(f"Message '{message}' published to topic '{topic}'")
            
            # Show the captured image with prediction
            cv2.imshow("Captured Image", image)
            cv2.waitKey(0)  # Wait for a key press to close the image
            cv2.destroyAllWindows()
        
        elif key == ord('q'):  # Press 'q' to quit
                        # Disconnect the client
            client.disconnect()
            print("Exiting program.")
            break

except KeyboardInterrupt:
    print("Program interrupted. Exiting...")
        # Disconnect the client
    client.disconnect()

finally:
        # Disconnect the client
    client.disconnect()
    picam2.stop()
    cv2.destroyAllWindows()
