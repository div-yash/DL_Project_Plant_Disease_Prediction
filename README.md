# 🌱 Plant Disease Classification using CNN

This project uses a **Convolutional Neural Network (CNN)** to classify plant leaf diseases into **38 categories** using the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset).

It includes:

* A **training notebook** (`plant_disease_training.ipynb`) for model building.
* A **Streamlit web app** (`main.py`) for real-time predictions.

---

## 📂 Project Structure

```
├── trained_model/
│   ├── plant_disease_prediction_model.h5   # Trained CNN model
│   ├── class_indices.json                  # Mapping of class indices to disease names
├── plant_disease_training.ipynb            # Jupyter Notebook (training pipeline)
├── main.py                                 # Streamlit app for inference
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## ⚙️ Features

* Dataset preprocessing with **Keras ImageDataGenerator**
* Custom **CNN model** for classification
* **Training & Validation** plots (accuracy, loss)
* **87% validation accuracy** achieved in 5 epochs
* Streamlit app for **real-time plant disease prediction**
* Top-3 predictions with confidence scores

---

## 📊 Dataset

* Source: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
* Number of classes: **38**
* Total images: \~54,000 (color set used)

---

## 🚀 Model Training (Notebook)

Run the notebook step by step:

1. **Install dependencies**

   ```bash
   pip install kaggle tensorflow matplotlib pillow
   ```
2. **Download dataset** (requires Kaggle API key)
3. **Preprocess images** (resize to 224×224, normalize)
4. **Build CNN model**:

   * Conv2D → MaxPooling → Conv2D → MaxPooling
   * Flatten → Dense(256) → Dense(38, Softmax)
5. **Train model**

   * Epochs: 5
   * Validation accuracy: \~87%
6. **Save model**

   ```python
   model.save("trained_model/plant_disease_prediction_model.h5")
   ```

---

## 🌐 Streamlit Web App (main.py)

Run the app locally:

```bash
pip install -r requirements.txt
streamlit run main.py
```

### Upload a leaf image

* Supported formats: `.jpg`, `.jpeg`, `.png`
* The app shows:

  * The uploaded image
  * Predicted disease class
  * Confidence score
  * Top-3 predictions

---

## 📸 Demo

Example prediction:

```
✅ Prediction: Apple___Black_rot (95.23%)

🔎 Top Predictions:
- Apple___Black_rot: 95.23%
- Apple___Cedar_apple_rust: 3.12%
- Apple___Apple_scab: 1.01%
```

---

## 📦 Requirements

Example `requirements.txt`:

```
tensorflow==2.17.0
numpy
pillow
matplotlib
streamlit
kaggle
```

---

## 🔮 Future Improvements

* Add **data augmentation** for better generalization
* Use **GlobalAveragePooling2D** instead of Flatten (reduce params)
* Try **transfer learning** (EfficientNet, ResNet) for higher accuracy
* Deploy on **HuggingFace Spaces / Streamlit Cloud**

---

## 🙌 Acknowledgements

* Dataset: [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
* Frameworks: TensorFlow, Keras, Streamlit
