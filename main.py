from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from supabase import create_client, Client
import numpy as np
import cv2
import io

app = FastAPI()

# Supabase setup
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_KEY = "YOUR_API_KEY"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load YOLO model
model = YOLO("yolov8n.pt")  # pastikan file model ini ada di folder Railway kamu nanti

@app.post("/deteksi")
async def deteksi(file: UploadFile = File(...)):
    # Baca gambar dari request
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Deteksi objek
    results = model(img)
    detections = results[0].boxes.data.tolist()

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = results[0].names[int(cls)]
        confidence = float(conf)

        if label.lower() == "bottle" and confidence > 0.5:
            print(f"🧴 Botol terdeteksi ({confidence:.2f}) — kirim ke Supabase...")
            data = {"user_id": "UUID_USER_KAMU", "jenis_sampah": "botol", "confidence": confidence}
            supabase.table("deteksi_sampah").insert(data).execute()
            return {"status": "berhasil", "label": label, "confidence": confidence}

    return {"status": "tidak ada botol terdeteksi"}
