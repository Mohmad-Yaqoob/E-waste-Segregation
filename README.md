# E-Waste Classification using Deep Learning

## 📌 Problem Statement
E-waste has become a significant environmental challenge due to the growing amount of discarded electronic devices. Manual classification of e-waste is inefficient and prone to human error. Therefore, automating the classification process using **Deep Learning** models can help in efficient sorting, recycling, and reducing environmental harm.

---

## 🎯 Objective
The goal of this project is to build a **robust image classification system** to classify e-waste images into their respective categories using different deep learning architectures. We compare **Custom CNN**, **InceptionV3**, and **ResNet152** to identify the most effective approach.

---

## ⚙️ Methodology

### 📂 Dataset
- Dataset consists of labeled e-waste images.
- Images were resized to **224x224 pixels** (standard size for CNN-based models like Inception and ResNet).
- Data augmentation techniques were applied to improve generalization and reduce overfitting:
  - Rotation
  - Flipping
  - Zoom
  - Shifting

### 🧠 Models Used
1. **Custom CNN**
   - Built from scratch with convolutional, pooling, and fully connected layers.
   - Served as a baseline model.

2. **InceptionV3**
   - A powerful deep learning model that uses **Inception modules** for efficient feature extraction.
   - Pretrained on ImageNet, then fine-tuned for e-waste dataset.

3. **ResNet152**
   - Uses **Residual Connections (skip connections)** to solve the vanishing gradient problem in deep networks.
   - Pretrained on ImageNet and fine-tuned.
   - Chosen because deeper networks can learn more complex representations.

---

## 🔧 Model Training & Hyperparameter Tuning
- **Image size**: 224x224 (to match pretrained model input requirements).
- **Optimizer**: Adam (adaptive learning rate for faster convergence).
- **Loss Function**: Categorical Crossentropy (multi-class classification).
- **Batch size**: 32
- **Epochs**: Tuned experimentally for best results.
- **EarlyStopping & ModelCheckpoint** were used to prevent overfitting and save best models.

---

## 📊 Results

| Model        | Accuracy | Precision | Recall | F1-Score |
|--------------|----------|-----------|--------|----------|
| Custom CNN   | 57%      | 0.55      | 0.57   | 0.56     |
| InceptionV3  | 96%      | 0.96      | 0.96   | 0.96     |
| ResNet152    | **99%**  | **1.00**  | **1.00** | **1.00** |

✅ **ResNet152 outperformed all other models**, making it the best choice for e-waste classification.

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone <https://github.com/Mohmad-Yaqoob/E-waste-Segregation>
   cd e-waste-classification
