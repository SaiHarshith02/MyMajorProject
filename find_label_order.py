"""
Run this locally to find the true label order from the trained model.
    python find_label_order.py
"""
import os
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

MODEL_PATH   = 'model.tflite'
DATA_DIR     = r'C:\Users\HARSHITH\OneDrive\Desktop\major project\BloodCancerClassification\Training'
IMAGE_SIZE   = 128
SAMPLES_EACH = 10   # images per class to test

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]['index']
out = interpreter.get_output_details()[0]['index']

def predict_index(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    arr = np.array(img, dtype=np.float32) / 255.0
    interpreter.set_tensor(inp, np.expand_dims(arr, 0))
    interpreter.invoke()
    return int(np.argmax(interpreter.get_tensor(out)[0]))

classes = sorted(os.listdir(DATA_DIR))
print(f"Classes on disk (sorted): {classes}\n")

votes = {}   # class_folder -> list of predicted indices

for cls in os.listdir(DATA_DIR):
    folder = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(folder):
        continue
    images = [f for f in os.listdir(folder)
              if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff'))][:SAMPLES_EACH]
    indices = [predict_index(os.path.join(folder, f)) for f in images]
    most_common = max(set(indices), key=indices.count)
    votes[cls] = most_common
    print(f"  '{cls}'  →  predicted index {most_common}  (raw: {indices})")

print("\n── Correct LABELS list for app.py ──")
label_map = [''] * 4
for cls, idx in votes.items():
    label_map[idx] = cls
print(f"LABELS = {label_map}")
