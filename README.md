# shoe-authentication-tool
# 👟 Shoe Authenticity Verification Web App

A Flask-based web application that uses a deep learning model to verify the authenticity of branded shoes based on multiple angle images (logo, side, top, sole, etc.).

---

## 🔧 Features

- Select Gender, Category, Brand, and Product.
- Upload images from required angles.
- Predicts model and angle using a trained Keras model.

---

## 🧠 Tech Stack

- Python 3.11
- Flask (Web framework)
- TensorFlow (Model inference)
- scikit-learn (LabelEncoder via `joblib`)
- Pillow (Image processing)
- NumPy

---

## 🚀 How to Run


```bash
git clone https://github.com/Dhanurdhar-Sharma/shoe-authentication-tool.git
cd shoe-authenticator
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py


## ⚠️ Python Compatibility

📌 **This app requires Python 3.11**  
TensorFlow does **not support Python 3.13** as of now.

To check your version:
```bash
python --version
if needed then download python 3.11


## 📁 Project Structure

.
├── app.py
├── requirements.txt
├── models/
│   └── brand_model.keras
│   └── brand_label_encoders.joblib
├── category/
│   └── shoes/
│        └── brand/
│           └── model/
│               └── model_summary.json
├── uploaded/
├── static/
│   └── demo_images/
├── templates/
│   ├── select_gender.html
│   ├── select_category.html
│   ├── select_brand.html
│   ├── select_product.html
│   ├── upload.html
│   └── result.html

Made with ❤️ by Dhanurdhar Sharma
is this correct
