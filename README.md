# ✋ Sign Language Recognition using CNN

This project implements an American Sign Language (ASL) recognition system using a **Convolutional Neural Network (CNN)**.  
It predicts hand gestures representing alphabets and supports real-time webcam recognition.

---

## 📂 Dataset
- **Source:** [ASL Alphabet Dataset - Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)  
- Contains **87,000+ images** of hand signs across **29 classes** (A–Z, SPACE, DELETE, NOTHING).  
- Images resized to **64×64 RGB** before training.

---

## 🧠 Model
- **Architecture:** Custom CNN (Convolutional Neural Network)  
- **Input size:** 64 × 64 × 3  
- **Output classes:** 29 (A–Z + special tokens)  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  

The trained model is saved as `SLR_final.h5`.  

⚠️ The model file is too large for GitHub (>100 MB).  
👉 Download it here: [Google Drive Link](https://drive.google.com/drive/folders/1JjK75ni5MON4727yQDLzLVElz4cRE7QY?usp=sharing)

---

## 📊 Results

### ✅ Accuracy
- **Training Accuracy:** 98.86%  
- **Validation Accuracy:** 95.89%  
- **Test Accuracy (Augmented):** 95.29%  

### ✅ Confusion Matrix
The confusion matrix shows the model’s performance across different classes:

![Confusion Matrix](Results/Confusion%20Matrix.png)


---

## 🎯 Sample Predictions

Here are some example outputs from the trained model:

| Sign | Prediction |
|------|------------|
| ![A](Results/A.png) | **A** |
| ![B](Results/B.png) | **B** |
| ![C](Results/C.png) | **C** |
| ![D](Results/D.png) | **D** |
| ![E](Results/E.png) | **E** |
| ![F](Results/F.png) | **F** |

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/hxrdikk/Sign-Language-Recognition.git
cd Sign-Language-Recognition
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Pretrained Model
Download `SLR_final.h5` from [Google Drive](https://drive.google.com/drive/folders/1JjK75ni5MON4727yQDLzLVElz4cRE7QY?usp=sharing)  
and place it in the **project root**.

### 4. Run Notebook
```bash
jupyter notebook SignLanguageRecognition.ipynb
```

### 5. Live Prediction via Webcam
```bash
python model_load.py
```

---

## 🏗 Project Structure
```
Sign-Language-Recognition/
├── Results/
│   ├── A.png
│   ├── B.png
│   ├── C.png
│   ├── D.png
│   ├── E.png
│   ├── F.png
│   └── Confusion_Matrix.png
├── model_load.py
├── test_Data_Split.py
├── requirements.txt
├── SignLanguageRecognition.ipynb
└── .gitignore
```

---

## 🏆 Features
- ASL alphabet recognition (A–Z + SPACE, DELETE, NOTHING)  
- Custom CNN architecture  
- Confusion matrix + classification report  
- Real-time webcam prediction with OpenCV  
- Dataset splitting script (`test_Data_Split.py`)  

---

## 👨‍💻 Authors
- **Hardik Jain**  
---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
