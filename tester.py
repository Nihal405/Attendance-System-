import cv2
import os
import numpy as np
import face_recognition
import datetime
import csv

# List of known face encodings and corresponding names
known_face_encodings = []
known_face_names = []

# Load images and encode faces
for image_file in os.listdir("resources"):
    image_path = os.path.join("resources", image_file)
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    name = os.path.splitext(image_file)[0] # Use filename as name
    known_face_encodings.append(encoding)
    known_face_names.append(name)

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Attendance dictionary to keep track of attendance duration for each person
attendance = {}
for name in known_face_names:
    attendance[name] = {'duration': 0, 'entry_time': None}

# Start time of the class and duration of the class
start_time = datetime.datetime.now()
class_duration = datetime.timedelta(minutes=15)

# Keep track of maximum attendance percentage seen so far
max_percentage = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Find faces in frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through faces in frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # Find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Calculate duration of face detection
            now = datetime.datetime.now()
            duration = (now - start_time).total_seconds()
            start_time = now

            # Add duration to attendance dictionary
            attendance[name]['duration'] += duration

            # Calculate percentage of attendance out of 15 minutes
            percentage = attendance[name]['duration'] / class_duration.total_seconds() * 100

            # Update maximum percentage seen so far
            if percentage > max_percentage:
                max_percentage = percentage
                max_name = name

            # Write entry time of the student in the attendance dictionary
            if attendance[name]['entry_time'] is None:
                attendance[name]['entry_time'] = now.strftime('%H:%M:%S')

        # Draw box around face and label with name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Attendance System', frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Write maximum attendance percentage and entry time of each student to attendance.csv file
with open("calculate.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Attendance Percentage", "Entry Time"])
    for name, data in attendance.items():
        percentage = data["duration"] / class_duration.total_seconds() * 100
        entry_time = data["entry_time"].strftime("%Y-%m-%d %H:%M:%S")
        print(name, percentage, entry_time)
        writer.writerow([name, f"{percentage:.2f}%", entry_time])


video_capture.release()
cv2.destroyAllWindows()
