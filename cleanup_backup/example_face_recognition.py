
# FACE RECOGNITION EXAMPLE
import face_recognition
import cv2

# Carica immagine di riferimento
reference_image = face_recognition.load_image_file("person.jpg")
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Riconosci in live video
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([reference_encoding], face_encoding)
        if True in matches:
            print("Persona riconosciuta!")
