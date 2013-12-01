import cv2, time
import urllib2, base64
import numpy as np

class ipCamera(object):

    def __init__(self,url, user = None, password = None):
        self.url = url
        auth_encoded = base64.encodestring('%s:%s' % (user, password))[:-1]

        self.req = urllib2.Request(self.url)
        self.req.add_header('Authorization', 'Basic %s' % auth_encoded)

    def get_frame(self):
        response = urllib2.urlopen(self.req)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, 1)
        return frame

class Camera(object):

    def __init__(self, camera = 0):
        self.cam = cv2.VideoCapture(camera)
        if not self.cam:
            raise Exception("Camera not accessible")
        self.shape = self.cam.read()[-1].shape

    def get_frame(self):
        if isinstance(self.shape, tuple):
            _,frame = self.cam.read()
        else:
            frame = np.zeros(480,640,3)
        return frame

    def release(self):
        self.cam.release()