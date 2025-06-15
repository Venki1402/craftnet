import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load("vehicle_classification_model.pth", map_location=torch.device('cpu')))
model.eval()

# Transformation for input image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("Vehicle Image Classification")
st.write("Upload an image of a vehicle (airplane, car, ship) to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_batch)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

    # Display image and result
    st.image(image, caption=f"Predicted: {predicted_class}", use_column_width=True)
    st.success(f"The predicted class is: {predicted_class}")
