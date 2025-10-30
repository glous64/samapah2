import os
import requests
from fastapi import FastAPI
from ultralytics import YOLO

app = FastAPI()

MODEL_PATH = "yolov8n.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
"

model = None  # model dimuat saat startup agar hemat RAM


def download_model():
    """Download model jika belum ada"""
    if not os.path.exists(MODEL_PATH):
        print("Downloading YOLO model...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded successfully!")


@app.on_event("startup")
def load_model():
    """Inisialisasi model YOLO saat aplikasi mulai"""
    global model
    download_model()
    model = YOLO(MODEL_PATH)
    print("YgitOLO model loaded and ready!")


@app.get("/")
def root():
    return {"message": "YOLO API is running!"}


@app.post("/predict")
def predict():
    """Contoh endpoint prediksi sederhana"""
    if model is None:
        return {"error": "Model not loaded."}
    # Contoh dummy response (tanpa upload gambar)
    return {"status": "ok", "message": "Prediction endpoint is working!"}
