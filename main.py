import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from flask import Flask, request, render_template_string

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report

# ==============================
# CONFIG
# ==============================
IMG_SIZE = 150
BATCH_SIZE = 32
MODEL_PATH = "model/pneumonia_model.h5"

# ==============================
# TRAIN MODEL (CNN)
# ==============================
def train_model():
    train_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_data = train_gen.flow_from_directory(
        'dataset/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_data = train_gen.flow_from_directory(
        'dataset/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, validation_data=val_data, epochs=10)

    # Save model
    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)

    # Plot graphs
    os.makedirs("screenshots", exist_ok=True)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.legend(['Train','Val'])
    plt.savefig("screenshots/accuracy.png")
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.legend(['Train','Val'])
    plt.savefig("screenshots/loss.png")

    print("✅ Training completed")

# ==============================
# EVALUATE MODEL
# ==============================
def evaluate_model():
    model = tf.keras.models.load_model(MODEL_PATH)

    test_gen = ImageDataGenerator(rescale=1./255)

    test_data = test_gen.flow_from_directory(
        'dataset/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    preds = model.predict(test_data)
    y_pred = (preds > 0.5).astype(int)
    y_true = test_data.classes

    cm = confusion_matrix(y_true, y_pred)

    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.savefig("screenshots/confusion_matrix.png")

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

# ==============================
# PREDICT SINGLE IMAGE
# ==============================
def predict_image(img_path):
    model = tf.keras.models.load_model(MODEL_PATH)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (150,150))
    img = img / 255.0
    img = np.reshape(img, (1,150,150,3))

    pred = model.predict(img)[0][0]
    return "PNEUMONIA" if pred > 0.5 else "NORMAL"

# ==============================
# FLASK WEB APP
# ==============================
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML = """
<h1>Pneumonia Detection (CNN)</h1>
<form method="POST" enctype="multipart/form-data">
    <input type="file" name="file"/>
    <input type="submit"/>
</form>
<h2>{{result}}</h2>
"""

@app.route("/", methods=["GET","POST"])
def index():
    result = ""
    if request.method == "POST":
        file = request.files["file"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        result = predict_image(path)
    return render_template_string(HTML, result=result)

# ==============================
# MAIN MENU
# ==============================
if __name__ == "__main__":
    print("""
1. Train Model
2. Evaluate Model
3. Run Web App
    """)
    choice = input("Enter choice: ")

    if choice == "1":
        train_model()
    elif choice == "2":
        evaluate_model()
    elif choice == "3":
        app.run(debug=True)
