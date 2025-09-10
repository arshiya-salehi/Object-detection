# Object Detection with HOG and Machine Learning

This project implements **object detection and face detection** using **Histogram of Oriented Gradients (HOG)** features and a classification pipeline. It demonstrates how to extract HOG features from images, train classifiers, and evaluate performance on test images.

---

## 📂 Project Structure

- **5_detection.ipynb** – Main Jupyter Notebook for training and testing detection models.
- **hogvis.py** – Utility for HOG feature visualization.
- **Datasets**:
  - `posX.jpg` / `aposX.jpg` – Positive training samples (objects/faces).
  - `negX.jpg` / `anegX.jpg` – Negative training samples (backgrounds).
  - `facesX.jpg` – Face examples for detection.
  - `testX.jpg` – Test images for evaluating detection.
- **Documentation**:
  - `5_detection.pdf` – Detailed report.
  - `5_detection(sub).pdf` – Supporting documentation.
  - `README.md` – Project overview.

---

## 🚀 Features

- Implementation of **Histogram of Oriented Gradients (HOG)** for feature extraction.
- Object and face classification using supervised learning methods.
- Visualization of HOG features for interpretability.
- Evaluation of detection accuracy on test images.

---

## 🛠️ Technologies Used

- **Python 3**
- **Jupyter Notebook**
- **NumPy**
- **scikit-image**
- **scikit-learn**
- **Matplotlib**

---

## 📖 How It Works

1. **Data Preparation**  
   - Positive and negative images are collected for training.  
   - Test images contain unseen faces/objects for evaluation.  

2. **Feature Extraction**  
   - HOG features are extracted from training images.  
   - Feature vectors are generated for classification.  

3. **Training**  
   - A classifier (e.g., SVM or similar model) is trained on the HOG features.  

4. **Detection**  
   - The trained model is applied to test images.  
   - Detected objects and faces are highlighted.

---

## 📊 Results

- Demonstrates successful object and face detection using HOG-based methods.
- Visualization of HOG gradients shows feature effectiveness.

---

## 📌 Future Work

- Integrate deep learning (e.g., CNNs) for improved accuracy.
- Expand dataset for better generalization.
- Real-time detection using webcam input.

---

## 👤 Author

This project was developed as part of an **Object Detection and Computer Vision assignment**.  
