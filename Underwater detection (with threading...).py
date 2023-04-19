import cv2
import threading
import numpy as np
from keras.models import load_model

# Load pre-trained model
model = load_model("D:/model/keras_model.h5")

# Define classes
classes = ['dolphin', 'fish', 'seahorse', 'shark', 'human', 'garbage']

# Define colors for bounding boxes
colors = [(0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 0, 255)]

# Initialize webcam capture
cap = cv2.VideoCapture(0)

def detect_objects():
    while True:
        # Read frame from webcam
        ret, frame = cap.read()

        # Preprocess frame for prediction
        img = cv2.resize(frame, (224, 224))
        img = np.array(img, dtype='float32')
        img = np.expand_dims(img, axis=0)

        # Make prediction
        pred = model.predict(img)

        # Get class with highest probability
        class_index = np.argmax(pred[0])
        label = classes[class_index]
        color = colors[class_index]

        # Draw bounding box and label on frame
        height, width, channels = frame.shape
        x1 = int(pred[0][1] * width)
        y1 = int(pred[0][0] * height)
        x2 = int(pred[0][3] * width)
        y2 = int(pred[0][2] * height)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,label,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

        # Show frame with bounding box and label
        cv2.imshow('Object Detection',frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start object detection thread
t = threading.Thread(target=detect_objects)
t.start()

# Wait for thread to finish
t.join()

# Release webcam capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
