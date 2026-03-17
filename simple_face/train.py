# train.py
import face_recognition
import pickle
import cv2
import os
from pathlib import Path

print("🔄 Loading and encoding faces from dataset/ ...")

known_encodings = []
known_names = []

dataset_path = "dataset"
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name) 
    if not os.path.isdir(person_folder):
        continue

    for image_path in Path(person_folder).glob("*.*"):
        print(f"  Processing: {person_name} - {image_path.name}")
        
        image = cv2.imread(str(image_path))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = face_recognition.face_locations(rgb_image, model="hog")
        encodings = face_recognition.face_encodings(rgb_image, boxes)
        
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(person_name)
        else:
            print(f"  ⚠️ No face found in {image_path.name}, skipping.")

data = {"encodings": known_encodings, "names": known_names}
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print(f"✅ Done! Encodings saved for {len(known_names)} faces.")