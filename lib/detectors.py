from openmdao.lib.datatypes.api import Float, Dict, Array, List, Int
from openmdao.main.api import Component, Assembly
import numpy as np
import cv2
import cv2.cv as cv


class cascadeDetection(Component):
    """
    Detects objects using pre-trained haar cascade files.
    
    Images should (at least ideally) be pre-grayscaled and contrast corrected,
    for best results.
    
    Outputs probable locations of these faces in an array with format:
    
    [[x pos, y pos, width, height], [x pos, y pos, width, height], ...]
    """
    
    def __init__(self, fn, scaleFactor = 1.3, minNeighbors = 4, 
                 minSize=(75, 75), flags = cv2.CASCADE_SCALE_IMAGE, 
                 persist = True, smooth = 10.):
        super(cascadeDetection,self).__init__()  
        self.add("frame_in", Array(iotype="in"))
        self.add("detected", Array(iotype="out"))
        self.scaleFactor = scaleFactor
        self.persist = persist # keep last detected locations vs overwrite with none
        self.minNeighbors = minNeighbors
        self.minSize = minSize
        self.flags = flags
        self.smooth = smooth
        self.cascade = cv2.CascadeClassifier(fn)
        self.find = True
        
        self.last_center = [0,0]
        
    def toggle(self):
        if self.find:
            self.find = False
        else:
            self.find = True
        return self.find
        
    def on(self):
        if not self.find:
            self.toggle()
    
    def off(self):
        if self.find:
            self.toggle()
    
    def shift(self,detected):
        x,y,w,h = detected
        center =  np.array([x+0.5*w,y+0.5*h])
        shift = np.linalg.norm(center - self.last_center)
        diag = np.sqrt(w**2 + h**2)
        
        self.last_center = center
        return shift
    
    def execute(self):
        if not self.find:
            return
        detected = self.cascade.detectMultiScale(self.frame_in, 
                                              scaleFactor=self.scaleFactor,
                                              minNeighbors=self.minNeighbors,
                                              minSize=self.minSize, 
                                              flags=self.flags)
        if not isinstance(detected,np.ndarray):
            return
        if self.smooth:
            if self.shift(detected[0]) < self.smooth: #regularizes against jitteryness
                return
        
        self.detected = detected
            
            


class faceDetector(cascadeDetection):
    """
    Detects forward-faces human faces in a frame.
    """

    def __init__(self, minSize=(50, 50)):
        #fn = "cascades/haarcascade_frontalface_default.xml"
        fn="cascades/haarcascade_frontalface_alt.xml"
        #fn="cascades/haarcascade_frontalface_alt2.xml"
        #fn = "cascades/haarcascade_frontalface_alt_tree"
        super(faceDetector, self).__init__(fn, minSize = minSize)
        self.add("foreheads", Array(iotype="out"))
        

    def get_foreheads(self):
        fh_x = 0.5  
        fh_y = 0.18
        fh_w = 0.25
        fh_h = 0.15
        forh = []
        for rect in self.detected:
            x,y,w,h = rect
            x += w * fh_x
            y += h * fh_y
            w *= fh_w
            h *= fh_h

            x -= (w / 2.0)
            y -= (h / 2.0)

            forh.append([int(x),int(y),int(w),int(h)])
        self.foreheads = np.array(forh)
        
    def execute(self):
        super(faceDetector, self).execute()
        self.get_foreheads()

class eyesDetector(cascadeDetection):
    """
    Detects forward-faces human faces in a frame.
    """

    def __init__(self, minSize=(40, 40)):
        fn = "cascades/haarcascade_eye.xml"
        super(eyesDetector, self).__init__(fn, minSize = minSize)