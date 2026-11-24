# Handwritten Digit Classification — MLOps Summative

An end-to-end IMAGE classification pipeline (not tabular). Dataset: scikit-learn handwritten digits, 1,797 grayscale 8x8 pixel images in ten classes (0-9). The model is a custom multilayer perceptron (MLP). It is saved as `.pkl` and reused as the pretrained starting point during retraining.

## Live links (fill in after deployment)
- Cloud app: **ADD CLOUD APP URL**
- GitHub repository: **ADD GITHUB URL**
- YouTube demonstration (camera on): **ADD YOUTUBE URL**

## Set up locally

Python 3.11 (recommended). In the project directory:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.model
uvicorn src.api:app --reload
```

Open http://127.0.0.1:8000. The notebook is `notebook/digit_mlops.ipynb`, and API documentation is http://127.0.0.1:8000/docs. Samples for prediction are at `data/test/`; select e.g. `digit_3.png`.

## How the ML pipeline works
1. Data acquisition: `load_digits()` produces real 8x8 digit images; stratified 80/20 train/test split.
2. Preprocessing: grayscale PNG/JPEG resized to 8x8; pixels scaled from 0..255 to 0..16, then divided by 16. Training images have foreground strokes on a dark background.
3. Custom neural network: MLP (96,48), ReLU, Adam optimizer, L2 regularization, early stopping.
4. Held-out evaluation: accuracy, macro precision, macro recall, macro F1, log loss, confusion matrix.
5. Uploaded labeled images are stored as SQLite BLOBs in `storage/uploads.sqlite`, so the database contains image data and labels.
6. A manual retraining button uses the saved custom MLP weights (`warm_start=True`), combined original training and new image data, and writes a new model artifact. Test metrics are refreshed.
7. API routes: `/health`, `/predict`, `/upload`, `/retrain`, `/metrics`, `/insights`.

## API usage
The UI lets users predict single images, upload multiple images all sharing the selected label, and click Retrain. To upload multiple classes, upload each class in a separate batch. Images must depict bright strokes on black background, ideally similar to sample images. **Only upload correctly labeled images**. Do not train using files in `data/test/` (test leakage); use `data/train/` or newly collected labeled samples.

## Docker — 1, 2 and 3 containers behind Nginx

```bash
docker compose up -d --build --scale api=1
# Site http://localhost:8080
# Run Locust experiment; then:
docker compose up -d --scale api=2
docker compose restart nginx
# Run same Locust experiment; then:
docker compose up -d --scale api=3
docker compose restart nginx
# Run same Locust experiment.
```

**Important:** During experiments, use only prediction endpoints. **Before retraining, scale back to one container**, to avoid simultaneous retraining processes touching a shared model/SQLite. Models and uploads persist in Docker named volume `appdata`.

Run locally installed Locust from a different terminal:

```bash
mkdir results
locust -f locustfile.py --host http://localhost:8080 --headless -u 20 -r 5 -t 60s --csv results/one_container
# Repeat after each scale change:
locust -f locustfile.py --host http://localhost:8080 --headless -u 20 -r 5 -t 60s --csv results/two_containers
locust -f locustfile.py --host http://localhost:8080 --headless -u 20 -r 5 -t 60s --csv results/three_containers
```

Read `results/*_stats.csv`: mean, median, p95 response times, requests per second, errors. Compare these in this README, include screenshots in the recorded demo, and optionally save CSV files to GitHub. Do not invent benchmark numbers.

| Containers | Users | Avg ms | p95 ms | Requests/sec | Failures |
|---|---:|---:|---:|---:|---:|
| 1 | 20 | Run test | Run test | Run test | Run test |
| 2 | 20 | Run test | Run test | Run test | Run test |
| 3 | 20 | Run test | Run test | Run test | Run test |

## Cloud deployment — Railway (Docker)
1. Push repository to GitHub.
2. On Railway, create a project from the GitHub repo. Railway detects root Dockerfile.
3. Create a persistent Volume mounted at `/app/storage`; otherwise uploads and model changes may vanish on redeploy. Set `STORAGE_DIR=/app/storage` and `PORT=8000` if needed.
4. Generate public domain for the service, then open the URL. Check `/health`, `/metrics`, upload a correctly labeled training image, press Retrain, and refresh `/metrics` to verify version increases. 
5. **Cloud model evaluation evidence:** Save screenshot of `/metrics` after retrain (accuracy, precision, recall, F1, log loss) and the site's uptime. Note that live latency is measured via Locust, not classification accuracy.

## Initial offline model result

Initial held-out accuracy: **0.9611** (360 test images, before any user retraining). Other metrics are saved in `models/initial_metrics.json` and reproduced by the notebook. Results after uploading new images may differ.

## Dataset visualization story
- Lit pixels: average number of stroke-occupied pixels by digit; larger values indicate more visual stroke coverage.
- Mean brightness: average digit stroke intensity; shows classes with relatively brighter image content.
- Horizontal stroke center: weighted horizontal location of the ink; reveals classes concentrated to left/right.
Visualizations are computed from the actual image arrays, not arbitrary chart data.

## Final submission checklist (45 points)
- [ ] Camera-on YouTube demo: predict + upload + retrain (5).
- [ ] SQLite save + preprocessing + trained MLP reuse + retraining (10).
- [ ] Correct prediction of sample digit image (10).
- [ ] Jupyter notebook preprocessing, model, >=4 metrics, optimizations (10).
- [ ] UI, uptime, visualizations, cloud URL + Docker (10).
- [ ] Flood request tests: 1, 2, 3 containers, latency and response comparisons.
- [ ] README links and real Locust figures updated.
- [ ] First Canvas attempt: ZIP of repository. Second attempt: GitHub URL.

**Security/scale limitation:** No authentication is included; public demo buttons permit uploads/retraining. Avoid sharing sensitive data and use the demo only for this course. Horizontal scaling experiments should not run concurrently with retraining.
