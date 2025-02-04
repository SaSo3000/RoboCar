# Arrow Direction Recognition using Neural Network & MQTT Communication  

## Description  
This project implements an **arrow direction recognition system** using a **neural network** trained with PyTorch. The trained model classifies arrow images into four categories: **Left, Right, Forward, and Back**.  
A **Raspberry Pi with a camera module** is used for real-time image capture, and the recognized direction is published via **MQTT** for IoT applications.  

---

## Features  
✔ Neural network trained using PyTorch with four direction classes.  
✔ Image preprocessing using OpenCV (grayscale, resize, normalize).  
✔ Real-time image capture using Raspberry Pi's **Picamera2** module.  
✔ **MQTT integration** to publish predicted directions to an IoT network.  

---

## Project Structure  
📂 dataset/new/dataset/ ├── 📂 Back/ ├── 📂 Forward/ ├── 📂 Left/ ├── 📂 Right/ 📜 model3.pth # Trained PyTorch model 

📜 Model_Training.py # Model training script 

📜 image_recognition_robot.py # Live prediction script with MQTT communication


1. Train the Model
Run train_model.py to train a neural network on your dataset.

2. Run Real-Time Prediction
Use the trained model to classify live images from a Raspberry Pi camera:


python image_recognition.py
Press 'c' to capture an image and predict its direction.
Press 'q' to quit the program.


MQTT Communication :

Broker: broker.hivemq.com
Port: 1883
Topic: direction
Message: Publishes the predicted label (0=Left, 1=Right, 2=Forward, 3=Back)


Model Architecture :

Layer	Type	Neurons
Input	Fully Connected	128×128 = 16,384
Hidden 1	Fully Connected (ReLU)	512
Hidden 2	Fully Connected (ReLU)	128
Hidden 3	Fully Connected (ReLU)	64
Output	Fully Connected	4 (Left, Right, Forward, Back)
Loss Function: CrossEntropyLoss
Optimizer: Adam (lr=1e-6)
Epochs: 450


Example Prediction Output :

Predicted Direction: Right
Message '2' published to topic 'direction'
A live image is displayed with its predicted label.
