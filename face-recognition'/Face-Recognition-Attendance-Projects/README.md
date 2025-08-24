A Python-based Face Recognition Attendance System that automates attendance marking using facial recognition technology. It uses OpenCV and the face_recognition library to detect and recognize faces in real-time via a webcam.

##Features

1)etects and recognizes multiple faces in real-time
2)Automatically records attendance with Name, Date, and Time
3)Stores attendance in a CSV file (Attendance.csv)
4)Uses a training dataset (Training_images folder) for known faces
5)Prevents proxy attendance and reduces manual effort

##Tech Stack

1)Python
2)OpenCV
3)face_recognition
4)NumPy, Pandas

##How It Works

1)Collect images of individuals and place them in the Training_images folder.
2)Run the program:
  python main.py
3)The webcam will start and detect faces.
4)Recognized faces are compared with trained images.
5)Attendance is logged in Attendance.csv with timestamp.

##Project Structure

Face-Recognition-Attendance/
│
├── main.py                # Main program file
├── requirements.txt       # Dependencies
├── Training_images/       # Folder containing training images
├── Attendance.csv         # Output file with attendance records
└── README.md              # Project documentation

##Applications

1)Schools & Colleges (student attendance)
2)Offices (employee attendance)
3)Events & Conferences (participant check-in)

##Installation & Setup

1)Install dependencies:
  pip install -r requirements.txt

2)Run the program:
  python main.py

##Future Enhancements

1)GUI-based interface for easier use
2)Store attendance in a database (MySQL/SQLite)
3)Cloud integration for remote access



