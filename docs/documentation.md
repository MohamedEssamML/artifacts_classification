# Artifacts Classification Documentation

## Overview
This project implements an image classification system for identifying historical museum artifacts using a simplified ResNet-based Convolutional Neural Network (CNN). The system classifies artifacts into categories such as pottery, sculptures, and textiles. A Flask web interface allows users to upload images and view classification results.

## Model Architecture
- **Model**: Simplified ResNet-inspired CNN
- **Input**: 224x224x3 RGB images
- **Architecture**:
  - Initial Conv2D (64 filters, 7x7, stride 2), BatchNormalization, ReLU, MaxPooling
  - 2 residual blocks (each with 2 Conv2D layers, 64 filters, 3x3)
  - GlobalAveragePooling, Dense(512), Dropout(0.5), Dense(num_classes, softmax)
- **Output**: Probability distribution over artifact categories
- **Loss**: Categorical cross-entropy
- **Optimizer**: Adam

## Dataset Preparation
- **Format**: Directory structure with subfolders for each category:
  ```
  dataset/
  ├── pottery/
  ├── sculptures/
  ├── textiles/
  └── ...
  ```
- **Preprocessing**:
  - Resize images to 224x224.
  - Normalize pixel values to [0, 1].
  - Augment data (rotation, flip, zoom) for robustness.
- **Recommended Datasets**: Custom museum artifact datasets or open-source datasets like those from museum APIs.

## Training
1. Prepare a labeled dataset as described above.
2. Run the training script:
   ```bash
   python train_classifier.py --dataset path/to/dataset
   ```
3. The script:
   - Loads and preprocesses the dataset.
   - Trains the CNN model.
   - Saves the trained model to `models/artifact_classifier.h5`.

## Inference
- **Script (`classify_artifact.py`)**:
  - Loads the pretrained model.
  - Preprocesses the input image.
  - Outputs the predicted artifact category.
- **Web Interface (`app.py`)**:
  - Upload an image via the Flask interface.
  - Displays the predicted category and confidence.

## Deployment
1. Install dependencies (`requirements.txt`).
2. Place the pretrained model in `models/`.
3. Run `app.py` to start the Flask server.
4. Access at `http://localhost:5000`.

## Implementation Details
- **Training (`train_classifier.py`)**:
  - Uses TensorFlow's `ImageDataGenerator` for data loading and augmentation.
  - Supports multi-class classification.
- **Inference (`classify_artifact.py`)**:
  - Preprocesses images to match training conditions.
  - Returns the top predicted class and confidence.
- **Web Interface (`app.py`)**:
  - Flask application with routes for uploading images and displaying results.

## Future Improvements
- Use a full ResNet50/101 for better accuracy.
- Incorporate transfer learning with pretrained models (e.g., ImageNet weights).
- Add support for multi-label classification.
- Integrate with museum databases for real-time artifact lookup.

## References
- He, K., et al. (2016). Deep Residual Learning for Image Recognition.
- Chollet, F. (2017). Deep Learning with Python.