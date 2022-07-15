import cv2
import numpy as np


img = cv2.imread("Images/environment.jpeg")

img2 = img.copy()

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cv2.imshow("Image2", img2)
cv2.waitKey(0)

# b,g,r = cv2.split(img)
# img2 = cv2.merge((r,g,b))
# pixel = img[100,100]
# print(pixel)

# print(img.shape)
# print("Number of rows: ", img.shape[0])
# print("Number of columns: ", img.shape[1])
# print(img.size)
# print(img.dtype)


# imgResize = cv2.resize(img, (600, 400), cv2.INTER_CUBIC)
# cv2.imshow("Resized Image", imgResize)
# cv2.waitKey(0)

# imgCrop = img[100:200, 200:400]
# cv2.imshow("Cropped Image", imgCrop)
# cv2.waitKey(0)


'''
pts1 = np.float32([[50, 50],[200, 50],[50, 200]])
pts2 = np.float32([[10, 100],[200, 50],[100, 250]])
M = cv2.getAffineTransform(pts1, pts2)
for pt in pts1:
    cv2.circle(img, tuple(pt), 5, (0, 255, 255), cv2.FILLED)
newImg = cv2.warpAffine(img, M, (512, 512))
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.imshow("New Image", newImg)
cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyWindow("Image")
'''

'''
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 530)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)

while True:
    success, img = cap.read()
    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("../Images/Test.png", img)
cv2.destroyAllWindows()
'''

'''
imgWidth = 640
imgHeight = 480
img = np.zeros((imgHeight, imgWidth, 3), np.uint8)

cv2.line(img, (0,0), (100,400), (0,255,0), 3)
cv2.rectangle(img, (100, 50), (200, 100), (255, 0, 0), 2)

cv2.imshow("image", img)
cv2.waitKey(0)
'''
