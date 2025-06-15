# main.py
from fastapi import FastAPI, File, UploadFile
import uvicorn
import torch
import tempfile
import cv2
import numpy as np
import torch.nn as nn

app = FastAPI()

# Step 1: Define the same model class used during training
class SignFormer(nn.Module):
    def __init__(self):
        super(SignFormer, self).__init__()
        # Define your model layers (update to match your training)
        self.linear = nn.Linear(512, 100)  # Replace this with your actual architecture

    def forward(self, x):
        return self.linear(x)

# Step 2: Load the model
model = SignFormer()
model.load_state_dict(torch.load("signformer_checkpoint.pth", map_location="cpu"))
model.eval()

# Dummy preprocessing & decoding (replace with real ones)
def preprocess(frames):
    # Resize, normalize, convert to tensor
    frames = np.stack(frames, axis=0)
    frames = frames.astype(np.float32) / 255.0
    return torch.from_numpy(frames).mean(dim=[0, 2, 3])  # dummy average frame vector

def decode_output(output):
    return f"Predicted ID: {output.argmax().item()} (replace with real translation)"

def video_to_frames(video_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()
    return frames

@app.post("/translate")
async def translate(video: UploadFile = File(...)):
    try:
        video_bytes = await video.read()
        frames = video_to_frames(video_bytes)
        input_tensor = preprocess(frames).unsqueeze(0)  # Add batch dim
        output = model(input_tensor)
        result = decode_output(output)
        return {"translation": result}
    except Exception as e:
        return {"error": str(e)}

# Run the API
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
