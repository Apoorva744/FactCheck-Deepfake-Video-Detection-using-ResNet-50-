# FactCheck - Deepfake Video Detection Using ResNet-50

A hybrid machine learning system that combines deep learning features with handcrafted computer vision techniques to detect manipulated videos with high accuracy.

## Overview

FactCheck leverages ResNet-50 for deep feature extraction combined with traditional computer vision features (LBP, edge detection, frequency analysis, color statistics) to identify subtle irregularities and artifacts in deepfake videos. The system processes video frames through a comprehensive pipeline involving feature extraction, scaling, and ensemble classification to achieve robust detection performance.

## Technical Architecture

The system implements a hybrid approach combining:
- **Deep Features**: ResNet-50 CNN extracts 2048-dimensional feature vectors per frame
- **Handcrafted Features**: Traditional computer vision techniques extract 519 additional features:
  - Multi-scale Local Binary Patterns (LBP): 260 features
  - Sobel edge detection histograms: 100 features
  - FFT frequency-domain features: 100 features
  - Color intensity statistics: 59 features
- **Total Feature Dimension**: 2567 features per frame
- **Classification**: Dual ensemble approach using Random Forest and XGBoost with probability calibration

## Performance Metrics

| Metric | Random Forest | XGBoost |
|--------|--------------|---------|
| Accuracy | 83.15% | 83.88% |
| ROC-AUC | 0.90 | 0.90 |
| F1-Score (Fake) | 0.84 | 0.84 |
| Precision (Fake) | 81% | 81% |
| Recall (Fake) | 87% | 88% |

**Detection Threshold**: 1% (FAKE_THRESHOLD = 0.01)  
**Test Dataset Size**: 2,214 videos

## Tech Stack

### Backend Technologies
- **Python 3.13.6**: Core programming language
- **TensorFlow/Keras**: ResNet-50 feature extraction with pre-trained ImageNet weights
- **OpenCV (cv2)**: Video frame extraction and preprocessing
- **Scikit-Learn**: StandardScaler for feature normalization, model evaluation metrics
- **XGBoost**: Gradient boosting classifier with tree pruning and regularization
- **Flask**: REST API backend for video upload and prediction endpoints
- **PostgreSQL**: Relational database for storing analysis results
- **NumPy**: Numerical array operations and feature vector storage
- **Joblib**: Model serialization and deserialization
- **Scikit-Image**: Handcrafted feature extraction (LBP, edge detection)

### Frontend Technologies
- **React.js**: Component-based UI framework
- **Chart.js & react-chartjs-2**: Timeline visualization and confidence score graphs
- **React Router**: Client-side routing
- **Node.js & npm**: Package management and build tools

## System Requirements

### Hardware Specifications
- **Processor**: AMD Ryzen 5 5500U or equivalent multi-core processor
- **RAM**: 16 GB DDR4 (minimum)
- **Storage**: 256 GB SSD (minimum)
- **Graphics**: Integrated Radeon Graphics or dedicated GPU (optional for acceleration)

### Software Requirements
- **Operating System**: Windows 11 / Windows 10 / Linux
- **Python**: 3.13.6 or higher
- **Node.js**: 14.x or higher
- **PostgreSQL**: Latest stable version
- **IDE**: VS Code or equivalent

## Installation

### Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install tensorflow==2.13.0
pip install opencv-python==4.8.1
pip install scikit-learn==1.3.2
pip install xgboost==2.0.3
pip install flask==3.0.0
pip install flask-cors==4.0.0
pip install psycopg2==2.9.9
pip install numpy==1.24.3
pip install joblib==1.3.2
pip install scikit-image==0.22.0
```

### Frontend Setup

```bash
cd frontend
npm install
npm install chart.js react-chartjs-2 react-router-dom
```

### Database Configuration

```sql
CREATE DATABASE factcheck_db;

CREATE TABLE videos (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    filename VARCHAR(255),
    prediction VARCHAR(10),
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Model Files

Ensure the following trained model files are placed in the `data/` directory:
- `calibrated_classifier.pkl`: Trained XGBoost/Random Forest with CalibratedClassifierCV wrapper
- `scaler.pkl`: Fitted StandardScaler for feature normalization

## Usage

### Starting the Application

**Backend Server:**
```bash
python app.py
```
Runs on `http://localhost:5000`

**Frontend Server:**
```bash
cd frontend
npm start
```
Opens at `http://localhost:3000`

### API Endpoints

**POST /api/predict**
- **Description**: Upload video for deepfake detection
- **Content-Type**: multipart/form-data
- **Parameters**: video file (mp4, avi, mov, mkv, webm)
- **Response**: JSON with prediction, confidence score, and timeline data

## Processing Pipeline

### 1. Frame Extraction
- OpenCV extracts uniformly sampled frames (max 50 per video)
- Temporal consistency maintained through even spacing
- BGR to RGB color space conversion

### 2. Preprocessing
- Frame resizing: 224×224 pixels (ResNet-50 input requirement)
- Pixel normalization: StandardScaler normalization
- Color format standardization: RGB

### 3. Feature Extraction

**Deep Features (ResNet-50):**
- Pre-trained on ImageNet dataset
- Final classification layer removed (include_top=False)
- Global Average Pooling applied
- Output: 2048-dimensional feature vector per frame

**Handcrafted Features:**
- **LBP (260 dims)**: Multi-scale texture patterns at radii 1, 2, 3, 4
- **Edge Features (100 dims)**: Sobel gradient magnitude and direction histograms
- **Frequency Features (100 dims)**: FFT-based spectral analysis
- **Color Features (59 dims)**: RGB channel statistics (mean, std, percentiles)

### 4. Feature Scaling
- StandardScaler normalization: zero mean, unit variance
- Ensures uniform feature importance during classification

### 5. Classification
- **Input**: 2567-dimensional feature vectors
- **Models**: Random Forest (200 trees) and XGBoost (200 estimators)
- **Calibration**: CalibratedClassifierCV for reliable probability estimates
- **Output**: Binary classification (REAL/FAKE) with confidence score

## Technical Justifications

### Why TensorFlow?
TensorFlow was selected for its production-ready deployment capabilities, optimized static computation graphs for inference-only workloads, and seamless Keras API integration with scikit-learn pipelines. Framework consistency ensures the calibrated classifier maintains feature distribution alignment, preventing accuracy degradation that would occur from framework switching.

### Why XGBoost?
XGBoost efficiently handles high-dimensional feature spaces (2567 dimensions) through L1/L2 regularization and tree pruning mechanisms. Its sequential gradient boosting corrects residual errors from previous iterations, achieving superior accuracy compared to parallel ensemble methods like Random Forest. Built-in class weighting addresses dataset imbalance more effectively than kernel-based methods (SVM) or linear models (Logistic Regression).

### Why Flask?
Flask's lightweight WSGI framework is optimal for single-purpose video processing APIs without the overhead of Django's ORM, admin interface, and templating engine. Its synchronous request handling aligns with CPU-bound video processing tasks, unlike FastAPI's async architecture designed for I/O-bound operations. Minimal boilerplate enables rapid prototyping while maintaining sufficient routing, file upload, and CORS capabilities for React integration.

## Datasets

### Training and Evaluation
1. **FaceForensics++ (C23 Compression)**: High-quality manipulated videos using multiple face-swap techniques (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
2. **Deepfake Detection (DFD) Dataset**: Google's large-scale collection with controlled real and manipulated video pairs

## Future Enhancements

1. **Audio-Visual Synchronization**: Detect lip-sync mismatches and voice manipulation
2. **Transformer-Based Models**: Implement Vision Transformers (ViT) and CLIP for improved feature representation
3. **Real-Time Detection**: Edge deployment with model quantization for mobile/browser-based verification
4. **Adversarial Robustness**: Continuous learning pipeline to adapt to emerging GAN architectures
5. **Blockchain Integration**: Immutable media provenance tracking for authenticity verification

## License

Academic research project for educational purposes.

---

**Note**: This system is designed for educational and research purposes. Production deployment should include additional security, scalability, and validation measures.
