import cv2
import numpy
import sys

"""
Measuring from 0,0 to the center of the eye looking straight ahead will be our
fixed measurement, from there we can take a picture assuming the subject hasnt
moved, we can take the measurement of the eyes new position and subtract it from
the fixed measurement to find how much it has moved and if it has which direction
by seeing if the total sum was a positive or negative number. This strategy can
be used on just the x or just the y axises to see how much the eye has moved up
and down or side to side.

Find the ceneter of the eye by using the w by h parameters and dividing those
numbers by half. PUT A PIXEL WHERE THE PREDICTED CENTER OF THE EYE IS
"""

# Get user supplied values
imagePath = "./abba.png"
cascPath = "./haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #cv2.putText(image,("x" + str(x) + "y" + str(y)), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    cv2.rectangle(image, (x+(w/2), y+(h/2)), (x+(w/2), y+(h/2)), (255, 0, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)