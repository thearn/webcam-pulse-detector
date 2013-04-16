import cv2

#Create object to read images from camera 0
cam = cv2.VideoCapture(0)

while True:
    #Get image from webcam and convert to greyscale
    ret, img = cam.read()

    #Display colour image with detected features
    cv2.imshow("Camera", img)

    #Sleep infinite loop for ~10ms
    #Exit if user presses <Esc>
    if cv2.waitKey(10) == 27:
        break
