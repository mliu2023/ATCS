import cv2
from cv2 import drawMarker
import numpy as np

# first image
img = cv2.imread("Images/pingry.jpeg")

# changing perspective
rows, cols, ch = img.shape

width, height = 300, 400
pts1 = np.float32([[250,0], [552,0], [243,366], [545, 366]])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
M = cv2.getPerspectiveTransform(pts1, pts2)
imgPts = img.copy()
for pt in pts1:
    cv2.circle(imgPts, (int(pt[0]),int(pt[1])), 5, (0, 255, 0), cv2.FILLED)
imgPersp = cv2.warpPerspective(imgPts, M, (width, height))

# sharpen
kernel = np.float32([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
imgSharpen = cv2.filter2D(imgPersp, -1, kernel)

cv2.imshow("Image with 4 Points", imgPts)
cv2.waitKey(0)
cv2.imshow("Perspective Transformation", imgPersp)
cv2.waitKey(0)
cv2.imshow("Sharpen", imgSharpen)
cv2.waitKey(0)

# second image
img2 = cv2.imread("Images/rainbowpallete.jpg")

# enlargen
img2 = cv2.resize(img2, (400,400))

# change hue
(h, s, v) = cv2.split(img2)
h = h + 30
h = np.clip(h,0,179)
s = np.clip(s,0,255)
v = np.clip(v,0,255)

imghsv = cv2.merge([h,s,v])
imghsv = cv2.cvtColor(imghsv, cv2.COLOR_BGR2HSV).astype("float32")
imgrgb = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)

# darken
darker = cv2.subtract(imgrgb, np.ones(imgrgb.shape, dtype = 'uint8')*40)

# sharpen
kernel = np.float32([[-3,-4,-3],[-4,29,-4],[-3,-4,-3]])
imgSharpen = cv2.filter2D(darker, -1, kernel)

environment = cv2.imread("Images/environment.jpeg")
environment = cv2.resize(environment, imgSharpen.shape[0:2])
added = cv2.addWeighted(environment, .5, imgSharpen, .5, 10)

cv2.imshow("Changed color channels", imgrgb)
cv2.waitKey(0)
cv2.imshow("Darker", darker)
cv2.waitKey(0)
cv2.imshow("Sharpened", imgSharpen)
cv2.waitKey(0)
cv2.imshow("Added", added)
cv2.waitKey(0)

# third image
img3 = cv2.imread("Images/nyc.jpeg")

# canny edge detection
imgGray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.blur(imgGray, (5,5))
imgCanny = cv2.Canny(imgBlur, 30, 220)

# increase saturation
(h, s, v) = cv2.split(img3)
s = s + 10
h = np.clip(h,0,179)
s = np.clip(s,0,255)
v = np.clip(v,0,255)

imghsv = cv2.merge([h,s,v])
imghsv = cv2.cvtColor(imghsv, cv2.COLOR_BGR2HSV).astype("float32")
imgrgb = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)

# adding the canny edge image to the image with increased saturation
combined = cv2.add(imgrgb, cv2.cvtColor(imgCanny, cv2.COLOR_GRAY2BGR))

cv2.imshow("NYC", img3)
cv2.waitKey(0)
cv2.imshow("Edges", imgCanny)
cv2.waitKey(0)
cv2.imshow("Changed saturation", imgrgb)
cv2.waitKey(0)
cv2.imshow("NYCEdges", combined)
cv2.waitKey(0)

# fourth image

# Add cascade to detect faces and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# ----- Get the faces and eyes in a static image
img4 = cv2.imread("Images/trumpbiden3.jpeg")

cv2.imshow("Trump and Biden", img4)
cv2.waitKey(0)

imgGray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

# Detect faces
# Scale: How much the image size is reduced at each image scale (default = 1.1)
# Neighbors: how many neighbors each candidate rectangle should have to retain it (default = 3)
faces = face_cascade.detectMultiScale(imgGray, 1.1, 4)

# Loop through all the faces and detect the eyes
for (x,y,w,h) in faces:
    # Add a bounding box around the face
    cv2.rectangle(img4, (x,y), (x+w,y+h), (255,0,0), 2)
    # Extract an image of the face only
    faceROI = imgGray[y:y+h, x:x+w]
    # Detect the eyes (eyes not detected)
    eyes = eye_cascade.detectMultiScale(faceROI)
    # Draw bounding boxes around the eyes
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img4,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
imgOpen = cv2.morphologyEx(img4, cv2.MORPH_OPEN, kernel)

# Show the result
cv2.imshow("Image w/ Faces", imgOpen)
cv2.waitKey(0)