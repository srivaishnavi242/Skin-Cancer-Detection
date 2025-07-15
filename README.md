# Skin Cancer Detection using 3D-TBP and Deep Learning

Skin cancer is one of the most prevalent cancers worldwide. Early and accurate detection is crucial for improving treatment outcomes. This project integrates 3D Total Body Photography (3D-TBP) and deep learning techniques to classify skin lesions as benign or malignant using dermoscopic images and metadata.

## 🔍 Project Overview

This repository presents three deep learning models developed and evaluated for skin cancer classification:
- Artificial Neural Network (ANN)
- VGG19 (Transfer Learning)
- Combined CNN-ANN Hybrid Model

Models are trained on the ISIC 2024 dataset and SLICE-3D lesion crops, featuring high-resolution images and metadata.

## 🚀 Features

- Classification of skin lesions as benign or malignant  
- 3D-TBP data integration using SLICE-3D  
- Multiple deep learning architectures with comparative analysis  
- Data augmentation and preprocessing pipeline  
- Performance visualization with confusion matrix, ROC-AUC, precision, recall, and F1-score  
- CLI and optional web interface (Streamlit/Flask) for inference

## 📂 Dataset

### ISIC Dataset
- Images: ~25,000 JPEG images  
- Metadata: Age, sex, lesion location, and diagnosis  
- Labels: Benign or Malignant

### SLICE-3D Dataset
- Cropped lesions from 3D-TBP images using Vectra WB360  
- Includes both histopathologically validated and clinically assumed labels

> **Note:** Please download the datasets manually from official sources and place them in the `/data` directory.

## 🏗️ Model Architectures

### ANN Model (3D Tensor-Based)
- Uses 3D convolutional layers for volumetric image processing  
- Dense layers for classification  
- Dropout and batch normalization for generalization

### VGG19 (Transfer Learning)
- Pretrained on ImageNet  
- Convolutional base frozen, custom dense layers trained for binary classification

### Combined CNN-ANN
- CNN for feature extraction  
- ANN for decision-making  
- Best overall performance with strong generalization

## 📊 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  
- Confusion Matrix

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/skin-cancer-3dtbp.git
cd skin-cancer-3dtbp

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download datasets and place in /data
```

## 🏋️‍♂️ Training

Train any model by specifying the architecture, number of epochs, batch size, and other parameters:

```bash
python train.py --model vgg19 --epochs 30 --batch-size 32
```


## 🧪 Evaluation

Evaluate a saved model on your test set:

```bash
python evaluate.py --model saved_model.h5 --test-dir data/test
```

## 📈 Results Summary

| Model         | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------|----------|-----------|--------|----------|---------|
| ANN           | 78%      | 0.76      | 0.77   | 0.76     | 0.85    |
| VGG19         | 85%      | 0.84      | 0.85   | 0.84     | 0.92    |
| Combined CNN  | 87%      | 0.86      | 0.87   | 0.86     | 0.93    |


## 🤝 Contributing

Contributions are welcome! If you'd like to improve the project, add a new model, enhance the interface, or extend functionality:

1. Fork the repository
2. Create a new branch  
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes  
   ```bash
   git commit -m 'Add some feature'
   ```
4. Push to the branch  
   ```bash
   git push origin feature-name
   ```
5. Open a Pull Request

---
