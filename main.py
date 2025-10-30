import os
import requests
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import shutil

app = FastAPI()

MODEL_PATH = "yolov8n.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
model = None


def download_model():
    """Download YOLO model jika belum ada"""
    if not os.path.exists(MODEL_PATH):
        print("Downloading YOLO model...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded successfully!")


@app.on_event("startup")
def load_model():
    """Load YOLO saat aplikasi mulai"""
    global model
    download_model()
    model = YOLO(MODEL_PATH)
    print("✅ YOLO model loaded successfully!")


@app.get("/")
def root():
    return {"message": "YOLO API is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded."}

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(temp_path)
    detections = results[0].boxes.xyxy.tolist()

    os.remove(temp_path)
    return {"detections": detections}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
