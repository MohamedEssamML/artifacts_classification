# Artifacts Classification inside Historical Museums

This project implements an image classification system to identify and categorize historical museum artifacts using a Convolutional Neural Network (CNN) based on a simplified ResNet architecture. A Flask-based web interface allows users to upload images of artifacts and receive classification results.

## Features
- Classify artifacts into categories (e.g., pottery, sculptures, textiles).
- Simplified ResNet-based CNN for robust image classification.
- Web interface for uploading and classifying artifact images.
- Modular scripts for training and inference.

## Requirements
- Python 3.8+
- TensorFlow 2.5+
- Flask
- NumPy
- OpenCV
- Pillow
- See `requirements.txt` for a complete list.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/artifacts_classification.git
   cd artifacts_classification
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure the `models/` directory contains the pretrained model (`artifact_classifier.h5`). Note: This is a placeholder; train the model or download pretrained weights.

## Usage
1. Start the Flask application:
   ```bash
   python app.py
   ```
2. Open a browser and navigate to `http://localhost:5000`.
3. Use the web interface to upload an artifact image and view the classification result.
4. Alternatively, run the classification script directly:
   ```bash
   python classify_artifact.py --image path/to/artifact.jpg
   ```
5. To train the model, prepare a dataset and run:
   ```bash
   python train_classifier.py --dataset path/to/dataset
   ```

## Project Structure
```
artifacts_classification/
├── app.py                    # Flask web application
├── train_classifier.py       # Training script
├── classify_artifact.py      # Inference script
├── models/                   # Pretrained model weights
├── static/                   # Static files (CSS, uploads)
├── templates/                # HTML templates
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── docs/                     # Detailed documentation
```

## Documentation
See `docs/documentation.md` for detailed information on the model architecture, dataset preparation, training, and deployment.

## Notes
- The pretrained model weight (`artifact_classifier.h5`) is a placeholder. Train the model using `train_classifier.py` with a labeled dataset.
- Dataset should contain images of artifacts organized by category (e.g., `dataset/pottery/`, `dataset/sculptures/`).
- This is a simplified implementation for educational purposes.
