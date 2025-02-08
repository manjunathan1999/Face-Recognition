import json
import os
import uuid
import cv2
import io, base64
from PIL import Image
import face_recognition as fr


SAVED = "./faces/"



class Face_recognition():

    def save_facial_image(base64_str,name):
        img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
        image_save = img.save(SAVED + name + ".jpg")
        return image_save


    def get_encoded_faces():
        encoded = {}
        for dirpath, dnames, fnames in os.walk(SAVED):
            for f in fnames:
                if f.endswith(".jpg") or f.endswith(".png"):
                    face = fr.load_image_file(SAVED + f)
                    encoding = fr.face_encodings(face)[0]
                    encoded[f.split(".")[0]] = encoding
        return encoded


    def unknown_image_encoded(img):
        face = fr.load_image_file(SAVED + img)
        encoding = fr.face_encodings(face)[0]
        return encoding


    def classify_face(rtsp_link):
        faces = Face_recognition.get_encoded_faces()
        known_face_encodings = list(faces.values())
        known_face_names = list(faces.keys())
        cap = cv2.VideoCapture(rtsp_link)

        while True:
            ret, frame = cap.read()
            # Find all face locations and encodings in the current frame
            face_locations = fr.face_locations(frame)
            face_encodings = fr.face_encodings(frame, face_locations)
            frame = cv2.resize(frame,(640,480))
            # Loop through each face found in the frame
            for face_encoding in face_encodings:
                # Compare the current face encoding with known faces
                matches = fr.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Check if there is a match
                if True in matches:
                    # Find the index of the first match
                    first_match_index = matches.index(True)
                    # Get the name corresponding to the match
                    name = known_face_names[first_match_index]
                    
                # Draw a rectangle around the face and display the name
                top, right, bottom, left = fr.face_locations(frame)[0]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
                
                output = {
                    "face_detection" : name
                }
                
            # Display the resulting frame
            cv2.imshow('Video', frame)

            # Check for the 'q' key to quit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture object and close the window
        cap.release()
        cv2.destroyAllWindows()