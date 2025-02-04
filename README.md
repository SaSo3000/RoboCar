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
