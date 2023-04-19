import cv2
import numpy as np
from tensorflow.keras.models import load_model
import threading

# Load the trained model
model = load_model("D:/model/keras_model.h5")

# Define the classes
classes = ['dolphin', 'fish', 'seahorse', 'shark', 'human', 'garbage']

# Set up the webcam
cap = cv2.VideoCapture(0)

# Initialize the bounding box list
bbox_list = []

# Define the thread function
def thread_function():
    global bbox_list
    while True:
        # Wait for a bounding box to be available
        while len(bbox_list) == 0:
            pass

        # Get the latest bounding box
        bbox = bbox_list[-1]

        # Draw a bounding box around the object
        x, y, w, h, class_name = bbox
        if class_name == 'garbage':
            color = (0, 0, 255)  # red
        else:
            color = (0, 255, 0)  # green
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness=2)

        # Display the class name
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, class_name, (x, y-10), font, 1, color, 2)

    # Create a new thread
thread = threading.Thread(target=thread_function)

# Start the thread
thread.start()

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Resize the frame to the size expected by the model
    resized_frame = cv2.resize(frame, (224, 224))

    # Normalize the pixel values to be between 0 and 1
    normalized_frame = resized_frame / 255.0

    # Reshape the frame to match the input shape of the model
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Make a prediction with the model
    predictions = model.predict(input_frame)

    # Get the indices of the classes with high probability
    class_indices = np.where(predictions[0] > 0.5)[0]

    # Loop over the detected classes and update the bounding box list
    for class_index in class_indices:
        class_name = classes[class_index]

        # Update the bounding box with the coordinates of the detected object
        mask = predictions[0] == predictions[0][class_index]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            bbox_list.append((x, y, w, h, class_name))

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
