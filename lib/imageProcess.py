from openmdao.lib.datatypes.api import Float, Dict, Array, List, Int
from openmdao.main.api import Component, Assembly
import numpy as np
import cv2
import cv2.cv as cv
from scipy import ndimage
from scipy import signal

class motionDiff(Component):
    diff_level = Float(iotype="out")

    def __init__(self):
        super(motionDiff,self).__init__()
        self.add("frame_in", Array(iotype="in"))

        self.add("diff", Array(iotype="out"))
        self.t0 = None
        self.t1 = None
        self.phase = None

    def diffImg(self,t0, t1, t2):
        d1 = cv2.absdiff(t2, t1)
        d2 = cv2.absdiff(t1, t0)
        """
        self.phase = np.array(cv2.phaseCorrelate(np.float32(self.t0[:,:,0]),
                                                 np.float32(self.t1[:,:,0])))
        self.phase = np.linalg.norm(self.phase)
        """
        t0 = np.float32(self.t0[:,:,0])
        t1 = np.float32(self.t1[:,:,0])
        self.phase = signal.fftconvolve(t0,t1)
        return cv2.bitwise_and(d1, d2)

    def execute(self):
        if not isinstance(self.t0, np.ndarray):
            self.t0 = self.frame_in
            self.diff = np.zeros(self.frame_in.shape)
            return
        if not isinstance(self.t1, np.ndarray):
            self.t1 = self.frame_in
            self.diff = np.zeros(self.frame_in.shape)
            return
        
        self.diff = self.diffImg(self.t0,self.t1,self.frame_in)
        self.t0 = self.t1
        self.t1 = self.frame_in
        self.diff_level = np.linalg.norm(self.diff)




class RGBSplit(Component):
    """
    Extract the red, green, and blue channels from an (n,m,3) shaped 
    array representing a single image frame with RGB color coding.

    Essentially a pretty straighforward numpy slicing operation.
    """

    def __init__(self):
        super(RGBSplit,self).__init__()
        self.add("frame_in", Array(iotype="in"))

        self.add("R", Array(iotype="out"))
        self.add("G", Array(iotype="out"))
        self.add("B", Array(iotype="out"))

    def execute(self):
        self.R = self.frame_in[:,:,0]     
        self.G = self.frame_in[:,:,1]   
        self.B = self.frame_in[:,:,2]   

class RGBmuxer(Component):
    """
    Take three (m,n) matrices of equal size and combine them into a single
    color r,g,b frame.
    """

    def __init__(self):
        super(RGBmuxer,self).__init__()
        self.add("R", Array(iotype="in"))
        self.add("G", Array(iotype="in"))
        self.add("B", Array(iotype="in"))

        self.add("frame_out", Array(iotype="out"))

    def execute(self):
        m,n = self.R.shape
        self.frame_out = cv2.merge([self.R,self.G,self.B])


class CVwrapped(Component):
    """
    Generic wrapper to take the simpler functions from the cv2 or scipy image
    libraries to generate connectable openMDAO components.

    The "simple" functions in mind are the ones of the form:

    "matrix in" --> [single method call]--> "matrix out"    

    Other functionality (like object detection, frame annotation, etc) should 
    probably be wrapped individually.
    """
    def __init__(self, func, *args, **kwargs):
        super(CVwrapped,self).__init__()
        self.add("frame_in", Array(iotype="in"))
        self.add("frame_out", Array(iotype="out"))
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def execute(self):
        self.frame_out = self._func(self.frame_in, *self._args, **self._kwargs)


class Grayscale(CVwrapped):
    """
    Turn (m,n,3) shaped RGB image frame to a (m,n) frame 
    Discards color information to produce simple image matrix.
    """
    def __init__(self):
        super(Grayscale,self).__init__(cv2.cvtColor, cv2.COLOR_BGR2GRAY)

class equalizeContrast(CVwrapped):
    """
    Automatic contrast correction.
    Note: Only works for grayscale images!
    """
    def __init__(self):
        super(equalizeContrast,self).__init__(cv2.equalizeHist)
        


if __name__ == "__main__":
    from device import Camera
    cam = Camera()
    proc = motionDiff()
    while True:
        frame = cam.get_frame()
        proc.frame_in = frame
        proc.run()
        
        diff = proc.diff
        
        print proc.phase
        
        cv2.imshow("test",diff)