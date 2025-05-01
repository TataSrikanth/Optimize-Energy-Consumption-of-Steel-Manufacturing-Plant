
# 🔍 Optimize Energy Consumption of Steel Manufacturing Plant

This is a **Streamlit-based web application** that allows users to upload a CSV file and get predictions using a **Support Vector Machine (SVM)** classification model.

---

## 📌 Features

- Upload CSV files containing specific features.
- Automatically preprocesses input data using scaling and encoding.
- Uses a trained SVM model (`svm_model.pkl`) to classify input samples.
- Appends predictions to the uploaded data.
- Allows users to **download the prediction results as a CSV file**.

---

## 🧠 Model Information

- Model type: `SVC` (Support Vector Classifier from `sklearn`)
- Preprocessing:
  - `StandardScaler` for input feature normalization.
  - `LabelEncoder` for target variable encoding and decoding.
- Trained and saved using `joblib`.

---

## 🧾 Input Requirements

Uploaded CSV files **must contain** the following columns:

- `ENERGY (Energy Consumption)`
- `Production (MT)`
- `TT_TIME (Total Cycle Time Including Breakdown)`

All rows with missing values in these columns will be dropped during prediction.

---

## 🚀 How to Run

### 1. Install Required Packages

```bash
pip install streamlit pandas scikit-learn joblib
```

### 2. Run the App

Make sure `svm_model.pkl` is in the same directory.

```bash
streamlit run app.py
```

---

## 📥 Example Usage

1. Launch the app.
2. Upload a CSV file with the required features.
3. View the predictions added to the dataset.
4. Download the updated CSV.

---

## 📂 Files

- `app.py` – Main Streamlit app.
- `svm_model.pkl` – Pre-trained SVM model.
- `final_code.ipynb` – Notebook used to develop/train the model. *(Optional)*

---

## 📬 Author

Developed by **@srikanth**  
For any quries contact:
Email : srikanthtata2002@gmail.com

