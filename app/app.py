# app.py
from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from model import train_profile, load_profile, apply_profile
from pathlib import Path

app = FastAPI()

PROFILE_PATH = "profiles/default.json"


@app.post("/train")
def train():
    profile = train_profile("data/training_imgs", PROFILE_PATH)
    return {"status": "trained", "profile_path": PROFILE_PATH}


@app.post("/apply")
async def apply(file: UploadFile = File(...)):
    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    profile = load_profile(PROFILE_PATH)
    result = apply_profile(img, profile)

    result = (result * 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    _, buffer = cv2.imencode(".jpg", result)

    return {
        "image": buffer.tobytes()
    }