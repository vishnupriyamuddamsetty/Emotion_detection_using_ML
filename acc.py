import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load model from JSON file
json_file = open('top_models\\fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load weights and attach them to the model
model.load_weights('top_models\\fer.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# Initialize variables to track accuracy
total_samples = 0
correct_predictions = 0

# Lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

while True:
    ret, img = cap.read()
    if not ret:
        break

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.1, 6, minSize=(150, 150))

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        predictions = model.predict(img_pixels)
        max_index = int(np.argmax(predictions))

        emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
        predicted_emotion = emotions[max_index]

        # Assume you have a true_label variable that holds the true emotion label
        true_label = 'happiness'  # Replace with your true label

        true_labels.append(true_label)
        predicted_labels.append(predicted_emotion)

        if true_label == predicted_emotion:
            correct_predictions += 1
        total_samples += 1

        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        resized_img = cv2.resize(img, (1000, 700))
        cv2.imshow('Facial Emotion Recognition', resized_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calculate accuracy
accuracy = (correct_predictions / total_samples) * 100


# Create a bar chart to represent accuracy
emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
emotion_counts = [true_labels.count(emotion) for emotion in emotions]
predicted_emotion_counts = [predicted_labels.count(emotion) for emotion in emotions]

plt.figure(figsize=(10, 6))
plt.bar(emotions, emotion_counts, label='True Labels')
plt.bar(emotions, predicted_emotion_counts, alpha=0.5, label='Predicted Labels')
plt.xlabel('Emotions')
plt.ylabel('Counts')
plt.legend()
plt.title('Emotion Recognition Accuracy')
plt.show()
