"""FastAPI: UI, prediction, labeled upload saved to SQLite, retraining, metrics."""
import io
import json
import os
import sqlite3
import time
from pathlib import Path
from threading import Lock
import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from src.model import STORAGE, MODEL_PATH, METRICS_PATH, initial_train, evaluate, save_model, save_metrics
from src.preprocessing import dataset, features, decode_image, explain_features
from src.prediction import predict

app = FastAPI(title='Digit Image MLOps')
START = time.monotonic()
DB_PATH = STORAGE / 'uploads.sqlite'
lock = Lock()


def db():
    STORAGE.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute("""CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        label INTEGER NOT NULL,
        image BLOB NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    return con


@app.on_event('startup')
def setup():
    # Docker image already contains pretrained artifact. This is fallback for local running.
    if not MODEL_PATH.exists():
        initial_train()
    with db() as con:
        pass


@app.get('/')
def home():
    return FileResponse('static/index.html')


@app.get('/health')
def health():
    return {'status': 'healthy', 'uptime_seconds': int(time.monotonic() - START),
            'model_loaded': MODEL_PATH.exists(), 'container': os.getenv('HOSTNAME', 'local')}


@app.post('/predict')
async def make_prediction(file: UploadFile = File(...)):
    try:
        return predict(await file.read())
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/upload')
async def upload_training_data(label: int = Form(...), files: list[UploadFile] = File(...)):
    if label not in range(10):
        raise HTTPException(status_code=400, detail='Label must be a digit 0 to 9.')
    if not files or len(files) > 100:
        raise HTTPException(status_code=400, detail='Upload 1–100 labeled images at a time.')
    rows = []
    for f in files:
        raw = await f.read()
        try:
            decode_image(raw)  # Verify before saving to database
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f'{f.filename}: {exc}') from exc
        rows.append((Path(f.filename or 'image.png').name, label, raw))
    with db() as con:
        con.executemany('INSERT INTO uploads(filename,label,image) VALUES(?,?,?)', rows)
    return {'saved_to_database': len(rows), 'label': label}


@app.post('/retrain')
def retrain():
    # A single container should be used during retraining; scale back to 1 first.
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail='Retraining is already running.')
    try:
        with db() as con:
            rows = con.execute('SELECT image, label FROM uploads').fetchall()
        if not rows:
            raise HTTPException(status_code=400, detail='Upload labeled images first.')
        X_extra = np.asarray([decode_image(raw) for raw, label in rows], dtype=np.float32)
        y_extra = np.asarray([label for raw, label in rows], dtype=int)
        X_train, X_test, y_train, y_test = dataset()
        model = joblib.load(MODEL_PATH)  # PREVIOUSLY TRAINED CUSTOM MODEL, NOT NEW MODEL
        before = evaluate(model, X_test, y_test)
        model.set_params(warm_start=True, early_stopping=False, max_iter=35)
        model.best_loss_ = float("inf")  # Required when switching from early stopping to continued training
        X_mix = np.concatenate([X_train, X_extra])
        y_mix = np.concatenate([y_train, y_extra])
        model.fit(features(X_mix), y_mix)
        after = evaluate(model, X_test, y_test)
        old_version = 1
        if METRICS_PATH.exists():
            old_version = json.loads(METRICS_PATH.read_text()).get('model_version', 1)
        save_model(model)
        after.update({'model_version': old_version + 1,
                      'retrained_images': len(rows), 'previous_accuracy': before['accuracy']})
        save_metrics(after)
        return after
    finally:
        lock.release()


@app.get('/metrics')
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail='Model not ready')
    # Evaluation performed against original held-out test set, also after retraining.
    return json.loads(METRICS_PATH.read_text())


@app.get('/insights')
def insights():
    X_train, X_test, y_train, y_test = dataset()
    imgs = np.concatenate([X_train, X_test])
    labels = np.concatenate([y_train, y_test])
    occupied, mean_pixel, cx = explain_features(imgs)
    return {'digits': list(range(10)),
            'counts': [int((labels == i).sum()) for i in range(10)],
            'mean_lit_pixels': [round(float(occupied[labels == i].mean()), 2) for i in range(10)],
            'mean_brightness': [round(float(mean_pixel[labels == i].mean()), 2) for i in range(10)],
            'horizontal_center': [round(float(cx[labels == i].mean()), 2) for i in range(10)],
            'explanations': [
                'Lit-pixel count measures how much of each image is occupied by strokes.',
                'Brightness measures average stroke intensity across the image.',
                'Horizontal center measures whether strokes tend to lie left or right.'
            ]}
