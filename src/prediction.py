"""Production prediction: re-read changed model artifact after retraining."""
from functools import lru_cache
import joblib
from src.model import MODEL_PATH
from src.preprocessing import decode_image, features

@lru_cache(maxsize=4)
def _load(path, modified_ns):
    return joblib.load(path)


def predict(raw):
    image = decode_image(raw)
    model = _load(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
    probabilities = model.predict_proba(features(image[None, :, :]))[0]
    best = int(probabilities.argmax())
    return {'digit': best, 'confidence': round(float(probabilities[best]), 4)}
