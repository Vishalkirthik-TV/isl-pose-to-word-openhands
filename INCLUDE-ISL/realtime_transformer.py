# Realtime Transformer Model Inference
# This script captures webcam frames, extracts keypoints, and runs inference using the trained transformer model.

import cv2
import numpy as np
import torch
from models.transformer import Transformer
from configs import TransformerConfig
from generate_keypoints import process_frame  # You may need to adapt this from process_video
import utils

# Load transformer model
config = TransformerConfig()
model = Transformer(config)
model.eval()

# Load pretrained weights (update path as needed)
model.load_state_dict(torch.load('path_to_pretrained_transformer.pth', map_location='cpu'))

# Start webcam
cap = cv2.VideoCapture(0)

print("Starting realtime detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract keypoints from the frame
    keypoints = process_frame(frame)  # You may need to implement this function
    if keypoints is None:
        continue

    # Prepare input for transformer (adapt as needed)
    input_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()

    # Display prediction
    cv2.putText(frame, f'Prediction: {pred}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Realtime Transformer Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
