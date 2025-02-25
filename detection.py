import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Image path for inference
image_path = 'dataset\close-up-dirt-ground-with-lot-.jpg'  # Replace with the path to your image

# Perform inference
results = model(image_path)

# Print detected objects
results.print()

# Show results
results.show()

# Save annotated image
results.save(save_dir='output')  # Save results in 'output' folder

# Display annotated image using OpenCV (optional)
img = cv2.imread(f'output/{image_path.split("/")[-1]}')
cv2.imshow('Detected Objects', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
