import cv2
from playsound import playsound
import time

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to detect drowsiness
def detect_drowsiness(frame):
    global sleepy_start_time
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(eyes) == 0:
            if time.time() - sleepy_start_time > 3:
                playsound("alarm_sound.mp3")  # Play alarm sound
                sleepy_start_time = time.time()  # Update start time of sleepy state
            cv2.putText(frame, "You are in sleepy danger :-(", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Check if both eyes are closed
            if len(eyes) < 2:
                if sleepy_start_time == 0:
                    sleepy_start_time = time.time()
            else:
                sleepy_start_time = 0  # Reset start time if eyes are open
                cv2.putText(frame, "Active :-)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Capture video from webcam
cap = cv2.VideoCapture(0)
sleepy_start_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_drowsiness(frame)

    cv2.imshow('Driver Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
