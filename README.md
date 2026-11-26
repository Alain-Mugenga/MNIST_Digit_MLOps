# Handwritten Digit Classification — MLOps
An end-to-end Machine Learning Operations (MLOps) project for classifying handwritten digits (0–9) using image data.

The project demonstrates image preprocessing, model training, evaluation, deployment, monitoring, and retraining through a web application.

## Live Application
**Web Application:** https://mnist-digit-mlops.onrender.com

**GitHub Repository:** https://github.com/Alain-Mugenga/MNIST_Digit_MLOps

**Video Demonstration:** https://youtu.be/D_A5UTFMdBc

## 1. Project Overview
This project uses the **scikit-learn Digits dataset**, containing 1,797 grayscale handwritten digit images of 8 × 8 pixels across ten classes (0–9).

A custom Multilayer Perceptron (MLP) classifier was developed using scikit-learn and deployed through a FastAPI application.

The system supports:

- Single-image digit prediction with confidence scores.
- Bulk uploading of labeled images to a SQLite database.
- Model retraining using previously trained model weights and newly uploaded images.
- Model evaluation and uptime monitoring.
- Dataset visualizations.
- Docker containerization and Locust performance testing.

**Technologies:** Python, scikit-learn, FastAPI, SQLite, Docker, Nginx, Locust, and Render.

## 2. Machine Learning Pipeline
### Data Acquisition and Preprocessing
The dataset is loaded using `sklearn.datasets.load_digits()` and split into training (80%) and testing (20%) sets using stratified sampling.

Each image is represented by 64 pixel features, normalized to values between 0 and 1.

Uploaded images are converted to grayscale, resized to 8 × 8 pixels, and normalized before prediction or retraining.

### Model Training
The classification model uses:

| Parameter | Configuration |
|---|---|
| Algorithm | Multilayer Perceptron (MLP) |
| Input Features | 64 |
| Hidden Layers | 96, 48 |
| Activation | ReLU |
| Optimizer | Adam |
| Optimization | L2 regularization and early stopping |
| Output Classes | 10 (digits 0–9) |
| Model Format | `.pkl` |

The trained model is saved in `models/digits_model.pkl`.

### Model Evaluation
The model was evaluated using 360 held-out test images.

| Metric | Initial Model | After Retraining |
|---|---:|---:|
| Accuracy | 96.11% | 97.22% |
| Precision (Macro) | 96.21% | 97.22% |
| Recall (Macro) | 96.06% | 97.19% |
| F1-score (Macro) | 96.04% | 97.19% |
| Log Loss | 0.1513 | 0.0838 |

Retraining was tested using three additional labeled images. The model version increased from 1 to 2, and accuracy improved by 1.11 percentage points.

These results were obtained on the scikit-learn Digits test dataset.

The detailed training, preprocessing, visualizations, and evaluation are documented in `notebook/digit_mlops.ipynb`.

## 3. Web Application
The application provides the following functionalities:

**Image Prediction:** Users upload a single digit image and receive the predicted class and confidence score.

**Bulk Image Upload:** Users upload multiple images belonging to a selected digit class. Images and labels are stored in SQLite for retraining.

**Model Retraining:** The retraining button loads the previously trained MLP, incorporates the newly uploaded images, continues training using `warm_start=True`, saves the updated model, and recalculates evaluation metrics.

**Monitoring:** Displays model uptime, health status, model version, and evaluation metrics.

### API Endpoints
| Endpoint | Function |
|---|---|
| `/health` | Health status and uptime |
| `/predict` | Image classification |
| `/upload` | Upload labeled images |
| `/retrain` | Trigger model retraining |
| `/metrics` | Model evaluation metrics |
| `/insights` | Dataset visualization data |

## 4. Dataset Visualizations
The application visualizes three image characteristics:

| Feature | Interpretation |
|---|---|
| Lit-Pixel Count | Measures stroke coverage. Digit 1 has fewer lit pixels on average than digits 0 and 8. |
| Mean Pixel Brightness | Measures average image intensity. Digit 8 has relatively higher brightness than digit 7. |
| Horizontal Stroke Center | Measures the horizontal distribution of strokes. Digits 3 and 9 have more right-shifted stroke centers than digit 5. |

These features demonstrate differences in handwriting patterns across digit classes.

Visualizations are available on the web application.

## 5. Installation and Local Setup
**Requirements:** Python 3.11 and pip.

Clone or download the repository and open the project directory.

### Create a Virtual Environment
```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

For macOS/Linux:

```bash
source .venv/bin/activate
```

### Install Dependencies
```bash
python -m pip install -r requirements-dev.txt
```

### Train the Model
```bash
python -m src.model
```

### Run the Application
```bash
uvicorn src.api:app --reload
```

Open:

- **Web UI:** http://127.0.0.1:8000
- **API Documentation:** http://127.0.0.1:8000/docs

## 6. Prediction and Retraining
### Prediction
1. Open the web application.

2. Navigate to **Predict one image**.

3. Upload an image from `data/test/`.

4. Click **Predict** to view the result.

Example prediction recorded on Render:

```json
{
  "digit": 6,
  "confidence": 0.9873
}
```

### Retraining
1. Navigate to **Upload new labeled images to database**.

2. Select the correct digit label (0–9).

3. Upload multiple corresponding images from `data/train/`.

4. Click **Upload images**.

5. Click **Retrain model**.

6. Refresh the evaluation metrics to view the updated results.

Images belonging to different digit classes should be uploaded in separate labeled batches.

The retraining process was successfully tested on Render, increasing the model version from 1 to 2.

## 7. Docker and Locust Performance Testing
Docker Compose and Nginx were used to test the application with 1, 2, and 3 API containers.

### Run with Docker
```bash
docker compose up -d --build --scale api=1
```

Access the application at http://localhost:8080.

Scale to two or three containers:

```bash
docker compose up -d --scale api=2
docker compose restart nginx
```

```bash
docker compose up -d --scale api=3
docker compose restart nginx
```

### Locust Load Testing
Locust simulated 20 concurrent users with a spawn rate of 5 users per second over 60 seconds.

Example command:

```bash
python -m locust -f locustfile.py --host http://localhost:8080 --headless -u 20 -r 5 -t 60s --csv results/one_container
```

The experiment was repeated with two and three containers.

### Performance Results
| Containers | Requests | Avg Response (ms) | Median (ms) | Max (ms) | Requests/sec | Failures |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3,059 | 60.10 | 54 | 738.90 | 52.73 | 0 |
| 2 | 3,027 | 64.38 | 55 | 1,423.81 | 52.74 | 0 |
| 3 | 3,093 | 57.11 | 55 | 273.72 | 53.80 | 0 |

**Interpretation:** All configurations processed requests without recorded failures. Three containers achieved the lowest average response time (57.11 ms) and highest throughput (53.80 requests/second) in these tests.

Scaling did not produce a consistent improvement at every step, as the two-container configuration had slightly higher average latency.

The experiments were conducted locally using Docker Compose. Detailed results are available in the `results/` directory.

## 8. Cloud Deployment
The application is deployed on **Render** using Docker and connected to the GitHub repository.

**Live URL:** https://mnist-digit-mlops.onrender.com

Deployment configuration:

| Setting | Value |
|---|---|
| Platform | Render |
| Service | Web Service |
| Runtime | Docker |
| Branch | `main` |
| Dockerfile | `Dockerfile` |
| Instance | Free |
| Auto-Deploy | On Commit |

The deployed application was tested successfully for image prediction, image uploads, model retraining, and production evaluation.

### Render Free Plan Limitations
Render Free instances automatically spin down after inactivity. The first request after inactivity may take approximately one minute or longer.

The free service also uses an ephemeral filesystem. Uploaded images, SQLite database changes, and retrained model files can be lost following restarts or redeployments.

Prediction and retraining can be demonstrated during the same active session. The repository retains the original trained model and application code.

Docker scaling experiments were performed locally, not on Render.

## 9. Model Limitations
The model was trained on **8 × 8 grayscale handwritten digits**, with bright strokes on dark backgrounds.

For reliable predictions, uploaded images should resemble the training dataset. Sample images are available in `data/test/`.

Although the application accepts other PNG or JPEG images, handwritten digits drawn in Paint or photographed from paper may be incorrectly classified due to differences in resolution, background color, stroke thickness, positioning, and handwriting style.

The reported accuracy applies to the held-out scikit-learn Digits test dataset and does not guarantee equivalent performance on unfamiliar handwriting.

Future improvements could include enhanced preprocessing and training with more diverse handwriting samples.

## 10. Project Structure
```text
MNIST_Digit_MLOps/
├── README.md
├── notebook/
│   └── digit_mlops.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   └── api.py
├── data/
│   ├── train/
│   └── test/
├── models/
│   ├── digits_model.pkl
│   └── initial_metrics.json
├── static/
│   └── index.html
├── results/
│   ├── final_comparison.csv
│   ├── one_container_stats.csv
│   ├── two_containers_stats.csv
│   └── three_containers_stats.csv
├── Dockerfile
├── docker-compose.yml
├── docker_start.sh
├── nginx.conf
├── locustfile.py
├── requirements.txt
└── requirements-dev.txt
```

## Conclusion
This project demonstrates an end-to-end MLOps pipeline for handwritten digit classification, covering model development, evaluation, deployment, monitoring, retraining, and performance testing.

The initial model achieved **96.11% accuracy**, increasing to **97.22%** after retraining in the tested session.

The application is publicly accessible through Render, while Docker and Locust experiments demonstrate container scaling and performance under concurrent requests.
