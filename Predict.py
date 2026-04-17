import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('model/pneumonia_model.h5')

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150,150))
    img = img / 255.0
    img = np.reshape(img, (1,150,150,3))

    prediction = model.predict(img)[0][0]

    return "PNEUMONIA" if prediction > 0.5 else "NORMAL"
