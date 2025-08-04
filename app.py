import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib

from flask import Flask, render_template, request, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = "b8e2f54ce750c3418c997e4025637e2c8f61ac4311fbc8a741726a7b4c980cfa"

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "category", "shoes")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploaded")
STATIC_DEMO_FOLDER = os.path.join(BASE_DIR, "static", "demo_images")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Utilities ---
def load_and_resize(img_path, target_size=(128, 128)):
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img).astype(np.float32) / 255.0
        assert img_array.shape == (128, 128, 3), f"Invalid shape: {img_array.shape}"
        return img_array
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

def get_brands(base_dir):
    return sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

def get_brand_models(base_dir, brand):
    brand_dir = os.path.join(base_dir, brand)
    products, summaries = [], []
    for folder in sorted(os.listdir(brand_dir)):
        summary_file = os.path.join(brand_dir, folder, "model_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f)
            products.append(summary["model"].strip())
            summaries.append((folder, summary))
    return products, [s for f, s in summaries]

def get_required_angles(summary):
    return summary.get("angles", [])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def select_gender():
    if request.method == "POST":
        session["gender"] = request.form.get("gender")
        return redirect(url_for("select_category"))
    return render_template("select_gender.html")

@app.route("/category", methods=["GET", "POST"])
def select_category():
    if request.method == "POST":
        session["category"] = request.form.get("category")
        return redirect(url_for("select_brand"))
    return render_template("select_category.html")

@app.route("/brand", methods=["GET", "POST"])
def select_brand():
    brands = get_brands(DATASET_DIR)
    if request.method == "POST":
        session["brand"] = request.form.get("brand")
        return redirect(url_for("select_product"))
    return render_template("select_brand.html", brands=brands)

@app.route("/product", methods=["GET", "POST"])
def select_product():
    brand = session.get("brand")
    products, summaries = get_brand_models(DATASET_DIR, brand)
    session["products"] = products
    session["summaries"] = summaries
    if request.method == "POST":
        session["product_idx"] = int(request.form.get("product_idx"))
        return redirect(url_for("upload_photos"))
    return render_template("select_product.html", products=products, brand=brand)

@app.route("/upload", methods=["GET", "POST"])
def upload_photos():
    brand = session.get("brand")
    prod_idx = session.get("product_idx")
    model_name = session["products"][prod_idx]
    summary = session["summaries"][prod_idx]
    angles = get_required_angles(summary)

    demo_imgs = [
        url_for("static", filename=f"demo_images/{brand}_{model_name.replace(' ', '_')}_{ang}.jpg")
        for ang in angles
    ]

    if request.method == "POST":
        files = {}
        for angle in angles:
            file = request.files.get(angle)
            if file and allowed_file(file.filename):
                filename = f"{brand}_{model_name}_{angle}_{file.filename}"
                path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(path)
                files[angle] = path
            else:
                flash(f"File for angle '{angle}' missing or invalid!", "danger")
                return redirect(request.url)
        session["uploaded_files"] = files
        return redirect(url_for("results"))

    return render_template("upload.html", brand=brand, model=model_name, angles=angles, demo_imgs=demo_imgs)

@app.route("/result")
def results():
    brand = session.get("brand")
    prod_idx = session.get("product_idx")
    model_name = session["products"][prod_idx]
    summary = session["summaries"][prod_idx]
    angles = get_required_angles(summary)
    files = session.get("uploaded_files", {})

    model_path = os.path.join(MODELS_DIR, f"{brand}_model.keras")
    encoder_path = os.path.join(MODELS_DIR, f"{brand}_label_encoders.joblib")

    if not (os.path.exists(model_path) and os.path.exists(encoder_path)):
        return "Model/encoder files not found. Please train first.", 500

    model = tf.keras.models.load_model(model_path)
    label_encoders = joblib.load(encoder_path)

    results = {}
    for angle in angles:
        fpath = files.get(angle)
        img = load_and_resize(fpath)
        if img is None:
            results[angle] = {"filename": os.path.basename(fpath), "status": "Error loading image"}
            continue
        img_batch = np.expand_dims(img, axis=0)
        preds = model.predict(img_batch)
        model_pred_id = np.argmax(preds[0], axis=1)[0]
        angle_pred_id = np.argmax(preds[1], axis=1)[0]
        pred_model = label_encoders["model"].inverse_transform([model_pred_id])[0]
        pred_angle = label_encoders["angle"].inverse_transform([angle_pred_id])[0]
        is_correct = (pred_model.strip().lower() == model_name.strip().lower() and pred_angle == angle)
        results[angle] = {
            "filename": os.path.basename(fpath),
            "pred_model": pred_model,
            "pred_angle": pred_angle,
            "expected_model": model_name,
            "expected_angle": angle,
            "status": "CORRECT" if is_correct else "MISMATCH!"
        }

    logo_status = results.get("logo", {}).get("status", "MISMATCH!")
    support_results = [v for k, v in results.items() if k != "logo"]
    mismatches = sum(1 for r in support_results if r["status"] != "CORRECT")

    if logo_status == "CORRECT" and mismatches == 0:
        verdict = "✅ PASS: Logo and all supporting images are correct."
    elif logo_status == "CORRECT":
        verdict = "✅ PASS (logo verified): but some support angles mismatched."
    else:
        verdict = "❌ FAIL: Logo mismatch. Cannot verify authenticity."

    return render_template("result.html", results=results, verdict=verdict, brand=brand, model=model_name)

if __name__ == "__main__":
    app.run(debug=True)
