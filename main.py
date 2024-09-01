import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
import cv2

# Create a directory to save gesture images
if not os.path.exists('gesture_images'):
    os.makedirs('gesture_images')

# List of hand gestures to scrape images for
gestures = ['thumbs_up', 'thumbs_down', 'peace', 'fist', 'okay', 'stop']

# Function to download images from Google
def download_gesture_images(gesture, num_images=5):
    url = f"https://www.google.com/search?q={gesture}+gesture&tbm=isch"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    images = soup.find_all('img')
    count = 0
    
    gesture_dir = f'gesture_images/{gesture}'
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)

    for img in images:
        if count >= num_images:
            break

        img_url = img['src']
        try:
            img_data = requests.get(img_url).content
            img_name = f"{gesture_dir}/{gesture}_{count}.jpg"
            with open(img_name, 'wb') as handler:
                handler.write(img_data)
            print(f"{gesture.capitalize()} image downloaded successfully.")
            count += 1
        except:
            continue

# Download 5 images per gesture (you can increase this)
for gesture in gestures:
    download_gesture_images(gesture)

# Resize images and preprocess them
img_size = (64, 64)  # Resize to 64x64 pixels

def preprocess_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Load and preprocess the dataset
images = []
labels = []

for i, gesture in enumerate(gestures):
    gesture_dir = f'gesture_images/{gesture}'
    for img_name in os.listdir(gesture_dir):
        img_path = os.path.join(gesture_dir, img_name)
        img_array = preprocess_image(img_path)
        images.append(img_array)
        labels.append(i)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Convert labels to one-hot encoding
labels = to_categorical(labels, num_classes=len(gestures))

# Define image augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Build the CNN model
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten and fully connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(gestures), activation='softmax'))  # Output layer for classification

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
model.fit(datagen.flow(images, labels, batch_size=16), epochs=10)

# Save the model
model.save('gesture_recognition_model.h5')

# Load the trained model
model = tf.keras.models.load_model('gesture_recognition_model.h5')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for prediction
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict the gesture
    prediction = model.predict(img)
    predicted_gesture = gestures[np.argmax(prediction)]

    # Draw an animated overlay
    overlay = frame.copy()
    cv2.putText(overlay, predicted_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    alpha = 0.5
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Display the prediction
    cv2.imshow('Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
