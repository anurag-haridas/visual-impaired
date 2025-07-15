# recognize_faces_live.py
import face_recognition
import cv2
import numpy as np
import pickle
import os
#from gtts import gTTS # Google Text-to-Speech
#from playsound import playsound # To play the audio file (install if you don't have it)
import pyttsx3 # For local Text-to-Speech
import threading
import time
# Initialize Text-to-Speech engine (pyttsx3 for local voice)
engine = pyttsx3.init()

# Optional: Configure voice properties (rate, volume, voice ID)
# Adjust speaking rate (words per minute) - default is usually 200
rate = engine.getProperty('rate')
engine.setProperty('rate', 150) # You can adjust this to make it slower or faster

# Adjust volume (0.0 to 1.0)
volume = engine.getProperty('volume')
engine.setProperty('volume', 0.9)

# Get available voices and set a preferred one
# This part is highly OS-dependent.
voices = engine.getProperty('voices')
# Example for Windows: Try to find a female voice (often 'Zira' or 'Hazel')
for voice in voices:
    # print(voice.id) # Uncomment to see available voice IDs
    if "zira" in voice.id.lower(): # Check for a specific voice name (Windows: Zira, David; Mac: Alex, Victoria)
        engine.setProperty('voice', voice.id)
        break
    elif "samantha" in voice.id.lower(): # MacOS default female voice
        engine.setProperty('voice', voice.id)
        break
    # For Linux, you might just use the default espeak voice or look for specific espeak voice IDs
    # e.g., 'english-us' for a generic US English voice

# If no specific voice is set, it will use the default system voice.

# Install gTTS and playsound:
# pip install gTTS playsound

ENCODINGS_FILE = "encodings.pkl" # File where you saved your encodings
SPEECH_DIR = "speech_audio" # Directory to store generated audio files

# Ensure the speech audio directory exists
if not os.path.exists(SPEECH_DIR):
    os.makedirs(SPEECH_DIR)

# Load known face encodings and names
try:
    with open(ENCODINGS_FILE, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    print(f"Loaded {len(known_face_names)} known faces.")
except FileNotFoundError:
    print(f"Error: {ENCODINGS_FILE} not found. Please run encode_faces.py first.")
    exit()

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
last_spoken_person = None
last_spoken_time = {} # To avoid repeating announcements too often

# Function to speak out text
# Global variables for cooldown (keep these as they are)
last_spoken_name = "None"
last_spoken_time = 0
SPEAK_COOLDOWN = 10 # seconds to wait before speaking the same name again, adjusted for potentially slower local TTS

def speak(text, name_for_cooldown=""):
    """
    Converts text to speech using pyttsx3 (local) and plays it.
    Uses a cooldown to prevent repeated announcements for the same name.
    """
    global last_spoken_name, last_spoken_time
    global engine # Access the initialized engine

    # Use a unique identifier for cooldown; if it's unknown, we still want to speak
    cooldown_id = name_for_cooldown if name_for_cooldown else "UNKNOWN"

    current_time = time.time()
    if cooldown_id == last_spoken_name and (current_time - last_spoken_time) < SPEAK_COOLDOWN:
        return # Don't speak if on cooldown for this name

    print(f"Speaking: {text}")
    try:
        engine.say(text)
        engine.runAndWait() # This command makes the engine speak and waits for it to finish

        last_spoken_name = cooldown_id
        last_spoken_time = current_time
    except Exception as e:
        print(f"Error speaking with pyttsx3: {e}")
    


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Starting real-time recognition. Press 'q' to quit.")

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to grab frame, exiting...")
        break

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6) # Adjust tolerance (lower is stricter)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            # Announce the person
            if name != "Unknown":
                speak(f"Hello, {name}", name)
            else:
                # Optionally announce "Unknown" if you want
                # speak("Unknown person detected", "Unknown")
                pass

    process_this_frame = not process_this_frame

    # Display the results (for visual debugging/monitoring)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()