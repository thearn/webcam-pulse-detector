import cv2, time

class Camera(object):

    def __init__(self, camera = 0):
        self.cam = cv2.VideoCapture(camera)
        if not self.cam:
            raise Exception("Camera not accessible")

        self.shape = self.get_frame().shape

    def get_frame(self):
        _,frame = self.cam.read()
        return frame
    
if __name__ == "__main__":
    import time
    import urllib
    import cv2
    cap = cv2.VideoCapture("test.asf")
    while True:
        ret,img = cap.read()
        print img
        if ret:
            cv2.imshow("Viewer",img)
