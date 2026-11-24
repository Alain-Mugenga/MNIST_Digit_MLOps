"""Initial model training, evaluation and example PNG creation."""
import json
import os
import tempfile
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
from src.preprocessing import dataset, features

STORAGE = Path(os.getenv('STORAGE_DIR', 'storage'))
MODEL_PATH = STORAGE / 'models' / 'digits_model.pkl'
METRICS_PATH = STORAGE / 'metrics.json'


def evaluate(model, X_test, y_test):
    preds = model.predict(features(X_test))
    probs = model.predict_proba(features(X_test))
    return {
        'accuracy': round(float(accuracy_score(y_test, preds)), 4),
        'precision_macro': round(float(precision_score(y_test, preds, average='macro', zero_division=0)), 4),
        'recall_macro': round(float(recall_score(y_test, preds, average='macro', zero_division=0)), 4),
        'f1_macro': round(float(f1_score(y_test, preds, average='macro', zero_division=0)), 4),
        'log_loss': round(float(log_loss(y_test, probs, labels=list(range(10)))), 4),
        'confusion_matrix': confusion_matrix(y_test, preds, labels=list(range(10))).tolist(),
        'test_samples': int(len(y_test)),
    }


def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=MODEL_PATH.parent, suffix='.pkl', delete=False) as f:
        temp_path = f.name
    try:
        joblib.dump(model, temp_path)
        os.replace(temp_path, MODEL_PATH)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def save_metrics(data):
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', dir=METRICS_PATH.parent, delete=False) as f:
        json.dump(data, f, indent=2)
        temp_path = f.name
    os.replace(temp_path, METRICS_PATH)


def initial_train():
    X_train, X_test, y_train, y_test = dataset()
    model = MLPClassifier(hidden_layer_sizes=(96, 48), activation='relu',
                          alpha=0.001, solver='adam', learning_rate_init=0.001,
                          early_stopping=True, n_iter_no_change=12,
                          max_iter=240, random_state=42)
    model.fit(features(X_train), y_train)
    save_model(model)
    results = evaluate(model, X_test, y_test)
    results['model_version'] = 1
    results['retrained_images'] = 0
    save_metrics(results)
    print('Model saved:', MODEL_PATH, 'Accuracy:', results['accuracy'])
    return results


def example_images():
    X_train, X_test, y_train, y_test = dataset()
    path = Path('data/test')
    path.mkdir(parents=True, exist_ok=True)
    for digit in range(10):
        candidates = np.where(y_test == digit)[0]
        predicted = joblib.load(MODEL_PATH).predict(features(X_test[candidates]))
        correct = candidates[predicted == digit]
        i = int(correct[0] if len(correct) else candidates[0])
        Image.fromarray((X_test[i] * 255 / 16).astype(np.uint8), mode='L').resize(
            (224, 224), Image.Resampling.NEAREST).save(path / f'digit_{digit}.png')
    path = Path('data/train')
    path.mkdir(parents=True, exist_ok=True)
    for digit in range(10):
        idx = np.where(y_train == digit)[0][:3]
        for j, i in enumerate(idx):
            Image.fromarray((X_train[i] * 255 / 16).astype(np.uint8), mode='L').resize(
                (224, 224), Image.Resampling.NEAREST).save(path / f'digit_{digit}_{j}.png')


if __name__ == '__main__':
    initial_train()
    example_images()
