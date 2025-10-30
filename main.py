import os
import requests
from ultralytics import YOLO
from fastapi import FastAPI

app = FastAPI()

MODEL_PATH = "yolov8n.pt"
MODEL_URL = "https://gtavpxkmbcfiakwkvwym.supabase.co/storage/v1/object/public/models/yolov8n.pt"

# Download model jika belum ada
if not os.path.exists(MODEL_PATH):
    print("Downloading YOLO model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Model downloaded successfully!")

# Load YOLO
model = YOLO(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "Server running with YOLO model!"}
