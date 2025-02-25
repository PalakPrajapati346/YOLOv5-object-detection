import torch
from yolov5 import YOLOv5
# Load YOLOv5 model
import pathlib 
temp=pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath
model = torch.hub.load('yolov5', 'custom', path='code\\best.pt', source="local", force_reload=True)
# Perform inference on an image
results = model('close-up-dirt-ground-wit-lot-.jpg')

# Print results
results.print()

# Show results
results.show()  # Displays the image with annotations
results.save(save_dir='output')
pathlib.PosixPath=temp
