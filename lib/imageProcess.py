from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp
import numpy as np
import time
import cv2

"""
Whole-frame image processing components & helper methods
"""

class RGBSplit(ExplicitComponent):

    """
    Extract the red, green, and blue channels from an (n,m,3) shaped
    array representing a single image frame with RGB color coding.

    At its core, a pretty straighforward numpy slicing operation.
    """

    def __init__(self):
        super(RGBSplit, self).__init__()
        self.add_input("frame_in", val=np.zeros((1, 1, 3)))
        self.add_output("R", val=np.zeros((1, 1)))
        self.add_output("G", val=np.zeros((1, 1)))
        self.add_output("B", val=np.zeros((1, 1)))

    def compute(self, inputs, outputs):
        frame_in = inputs['frame_in']
        outputs['R'] = frame_in[:, :, 0]
        outputs['G'] = frame_in[:, :, 1]
        outputs['B'] = frame_in[:, :, 2]


class RGBmuxer(ExplicitComponent):

    """
    Take three (m,n) matrices of equal size and combine them into a single
    RGB-coded color frame.
    """

    def __init__(self):
        super(RGBmuxer, self).__init__()
        self.add_input("R", val=np.zeros((1, 1)))
        self.add_input("G", val=np.zeros((1, 1)))
        self.add_input("B", val=np.zeros((1, 1)))
        self.add_output("frame_out", val=np.zeros((1, 1, 3)))

    def compute(self, inputs, outputs):
        R = inputs['R']
        G = inputs['G']
        B = inputs['B']
        outputs['frame_out'] = cv2.merge([R, G, B])


class CVwrapped(ExplicitComponent):

    """
    Generic wrapper to take the simpler functions from the cv2 or scipy image
    libraries to generate connectable openMDAO components for image processing.

    The "simple" functions in mind here are the ones of the form:

    "matrix in" --> [single method call]--> "matrix out"

    Other functionality (like object detection, frame annotation, etc) should
    probably be wrapped individually.
    """

    def __init__(self, func, *args, **kwargs):
        super(CVwrapped, self).__init__()
        self.add_input("frame_in", val=np.zeros((1, 1, 3)))
        self.add_output("frame_out", val=np.zeros((1, 1, 3)))
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def compute(self, inputs, outputs):
        frame_in = inputs['frame_in']
        outputs['frame_out'] = self._func(frame_in, *self._args, **self._kwargs)


class Grayscale(CVwrapped):

    """
    Turn (m,n,3) shaped RGB image frame to a (m,n) frame
    Discards color information to produce simple image matrix.
    """

    def __init__(self):
        super(Grayscale, self).__init__(cv2.cvtColor, cv2.COLOR_BGR2GRAY)


class equalizeContrast(CVwrapped):

    """
    Automatic contrast correction.
    Note: Only works for grayscale images!
    """

    def __init__(self):
        super(equalizeContrast, self).__init__(cv2.equalizeHist)


class showBPMtext(ExplicitComponent):

    """
    Shows the estimated BPM in the image frame
    """
    ready = False
    bpm = 0.0
    x = 0
    y = 0
    fps = 0.0
    size = 0.0
    n = 0

    def __init__(self):
        super(showBPMtext, self).__init__()
        self.add_input("frame_in", val=np.zeros((1, 1, 3)))
        self.add_output("frame_out", val=np.zeros((1, 1, 3)))
        self.bpms = []

    def compute(self, inputs, outputs):
        frame_in = inputs['frame_in']
        self.bpms.append([time.time(), self.bpm])
        if self.ready:
            col = (0, 255, 0)
            text = "%0.5f bpm" % self.bpm
            tsize = 2
        else:
            col = (100, 255, 100)
            gap = (self.n - self.size) / self.fps
            text = "(estimate: %0.5f bpm, wait %0.2f s)" % (self.bpm, gap)
            tsize = 1
        cv2.putText(frame_in, text,
                    (self.x, self.y), cv2.FONT_HERSHEY_PLAIN, tsize, col)
        outputs['frame_out'] = frame_in
