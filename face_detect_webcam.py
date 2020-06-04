import cv2
import numpy
import sys

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

image_path = 'Page Photo copy.png'
image = cv2.imread(image_path, 1)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        image_resize = cv2.resize(image, (w, h))

        roi_web_face = frame[y:y+h, x:x+w]
        roi_page_face = image_resize[0:h, 0:w]

        dst = cv2.add(roi_web_face, roi_page_face)
        frame[y:y+h, x:x+w] = dst
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.putText(frame,("x" + x + "y" +  y), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()