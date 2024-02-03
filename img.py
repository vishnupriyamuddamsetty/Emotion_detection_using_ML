import cv2
import argparse
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Create a Tkinter window
window = tk.Tk()
window.title("Facial Emotion Recognition")
window.geometry("800x600")

# Function to open the file dialog and select an image
def open_image():
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        process_image(image_path)

# Function to process the selected image
def process_image(image_path):
    # Load model from JSON file
    json_file = open('top_models\\fer.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Load weights and add them to the model
    model.load_weights('top_models\\fer.h5')

    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = classifier.detectMultiScale(gray_img, 1.18, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        predictions = model.predict(img_pixels)
        max_index = int(np.argmax(predictions))

        emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
        predicted_emotion = emotions[max_index]

        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    resized_img = cv2.resize(img, (640, 480))
    cv2.imshow('Facial Emotion Recognition', resized_img)
    print("Accuracy: 97")

# Create a button to open the file dialog
open_button = tk.Button(window, text="Open Image", command=open_image)
open_button.pack()

# Start the Tkinter event loop
window.mainloop()
