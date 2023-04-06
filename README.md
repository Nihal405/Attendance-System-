# Attendance-System-
In today's digital age, face recognition technology has become increasingly popular, particularly in security and surveillance applications. One such application is attendance tracking, which can be especially useful in schools, universities, and workplaces. This project demonstrates the development of a face recognition attendance system using OpenCV and face_recognition libraries in Python.

Working:
The system works by comparing the faces in a video feed to a database of known faces. The system captures video footage from a camera, detects faces within the frame using the OpenCV library, and then compares the detected faces to a pre-existing database of known faces using the face_recognition library. If a match is found, the system records the entry time of the individual and calculates their attendance percentage.

The system uses a dictionary to keep track of the attendance of each individual. If an individual is detected for the first time, their name is added to the dictionary along with their entry time. Subsequently, the duration of their attendance is recorded, and the attendance percentage is calculated by dividing the duration by the total duration of the class.

The system also has a graphical user interface that displays the real-time video feed with bounding boxes around detected faces and the names of the individuals. The system stores the attendance information in a CSV file, which can be easily exported to other applications.

Overall, this project provides a simple yet effective solution for tracking attendance using face recognition technology, which has the potential to save time and increase accuracy compared to traditional attendance tracking methods.
