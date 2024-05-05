Automated Entry Exit System for IIT G

https://github.com/Ritikkoshta02/Automated-Entry-Exit-System-using-Image-Processing/assets/78507349/922c02b7-70ad-40f7-8764-7ed7b904e386


Overview:

This project implements an Automated Entry Exit System for IIT G (Indian Institute of Technology, Guwahati) using image processing, face recognition, and text extraction techniques. It provides a seamless and efficient way to manage entry and exit of individuals based on their ID cards.

Features:

    ID Card Detection: Utilizes YOLO (You Only Look Once) to detect ID cards through a webcam in real-time.
    Image Processing: Processes and modifies the detected ID card image, aligning and cropping it for further analysis.
    Classification: Classifies the detected ID card into different types (e.g., College ID, Aadhar card, Driving license). Currently, implemented for College ID cards Model trained using CNN based Machine Learning Alogrithms.
    Face Recognition: Validates the identity of the ID holder by comparing the face on the ID card with a live picture captured using face recognition techniques.
    Text Extraction: Extracts text information from the ID card using pytesseract.
    Database Integration: Utilizes MySQL database for storing and retrieving ID card information and entry/exit logs.
    Frontend with Streamlit: Offers a user-friendly frontend powered by Streamlit for easy interaction and visualization of system functionalities.
    Automatic Logging: Records entry and exit information in an Excel sheet automatically based on even-odd logic.
    Notification System: Sends notifications to students outside the campus shortly before the gate closing time.

Installation:

    Clone the repository:

    bash
    git clone https://github.com/Ritikkoshta02/Automated-Entry-Exit-System-using-Image-Processing.git

Install dependencies:

bash

pip install -r requirements.txt

Set up MySQL database and configure database connection parameters in config.py.
Run the system:

bash

    streamlit run app.py

Usage

    Ensure the webcam is connected and functioning properly.
    Run the system and position the webcam to capture the ID cards as individuals approach the entry/exit point.

Future Scope

    Implementation of Aadhar card and Driving license recognition.
    Integration with a more comprehensive database system.
    Improvements in face recognition accuracy and speed.
    Enhancements in notification system for better user engagement.

Credits

    YOLO: YOLOv5: An Incremental Improvement
    Face Recognition: Face Recognition with OpenCV and Deep Learning
    pytesseract: Tesseract OCR
