#!/bin/sh
set -e
mkdir -p "$STORAGE_DIR/models"
if [ ! -f "$STORAGE_DIR/models/digits_model.pkl" ]; then
    cp /opt/initial_storage/models/digits_model.pkl "$STORAGE_DIR/models/digits_model.pkl"
fi
if [ ! -f "$STORAGE_DIR/metrics.json" ]; then
    cp /opt/initial_storage/metrics.json "$STORAGE_DIR/metrics.json"
fi
exec uvicorn src.api:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
