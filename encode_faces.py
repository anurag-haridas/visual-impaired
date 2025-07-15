# encode_faces.py
import face_recognition
import os
import pickle # To save the encodings efficiently

# Path to the directory containing known faces
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pkl" # File to save the facial encodings

known_face_encodings = []
known_face_names = []

print("Loading known faces...")

# Loop through each person's directory
for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if not os.path.isdir(person_dir):
        continue # Skip if it's not a directory

    print(f"Processing {name}...")
    for filename in os.listdir(person_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(person_dir, filename)
            try:
                # Load the image
                image = face_recognition.load_image_file(image_path)
                # Get face encodings (128-dimensional vector)
                # It can find multiple faces, so we take the first one found
                face_encodings = face_recognition.face_encodings(image)

                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(name)
                else:
                    print(f"Warning: No face found in {image_path}. Skipping.")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

# Save the encodings and names to a file
with open(ENCODINGS_FILE, 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print(f"Face encodings saved to {ENCODINGS_FILE}")
print(f"Total known faces encoded: {len(known_face_names)}")