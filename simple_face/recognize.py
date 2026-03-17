# recognize.py
import face_recognition
import cv2
import pickle
import time

print("📦 Loading encodings and starting camera...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())

known_encodings = data["encodings"]
known_names = data["names"]

# Initialize camera (Linux: /dev/video0 is usually index 0)
video_capture = cv2.VideoCapture(1)

# Cooldown to avoid logging same person repeatedly
last_seen = {}
cooldown_seconds = 3

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    current_time = time.time()

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_indices = [i for i, match in enumerate(matches) if match]
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = min(match_indices, key=lambda i: distances[i])
            name = known_names[best_match_index]

            if name not in last_seen or (current_time - last_seen[name] > cooldown_seconds):
                print(f"✅ [SECURITY LOG] {name} recognized at {time.ctime()}")
                # TODO: Log to database
                last_seen[name] = current_time
        else:
            print(f"⚠️ [SECURITY ALERT] Unknown face detected at {time.ctime()}")
            # TODO: Save unknown face image
            # cv2.imwrite(f"unknown_faces/{time.time()}.jpg", frame)

        # Draw on frame
        top, right, bottom, left = [coord * 4 for coord in face_location]
        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

    cv2.imshow('Security System - Face Recognition (Linux)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
print("👋 System stopped.")