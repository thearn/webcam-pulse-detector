from openmdao.lib.datatypes.api import Float, Dict, Array, List, Int
from openmdao.main.api import Component, Assembly
import numpy as np
import cv2

"""
Image frame analysis components written to operate only on inputted regions
(slices of numpy arrays) which are inputted. 

Typically these recieve, as input, the output of some object detection components
"""

class processRect(Component):
    """
    Process inputted rectangles, using specification 
    [ [x pos, y pos, width, height], ... ]
    into an inputted frame.
    
    (Used as a prototype for most of the region-specific image analysis 
    components)
    """
    
    def __init__(self, channels = [0,1,2], zerochannels = []):
        super(processRect,self).__init__()
        self.add("frame_in", Array(iotype="in"))
        self.add("rects_in", Array(iotype="in"))
        self.add("frame_out", Array(iotype="out"))
        self.channels = channels
        self.zerochannels = zerochannels
        
    def execute(self):
        temp = np.array(self.frame_in) # bugfix for strange cv2 error
        if self.rects_in.size > 0:
            for rect in self.rects_in:
                if len(self.frame_in.shape) == 3:
                    for chan in self.channels:
                        temp[:,:,chan] = self.process(rect, temp[:,:,chan])
                    x,y,w,h = rect
                    for chan in self.zerochannels:
                        temp[y:y+h,x:x+w,chan]= 0*temp[y:y+h,x:x+w,chan]
                else:
                    temp = self.process(rect, temp)
        self.frame_out = temp
        
    def process(self):
        return 

class drawRectangles(processRect):
    """
    Draws rectangles outlines in a specific region within the inputted frame
    """
        
    def process(self, rect, frame):
        x,y,w,h = rect
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
        return frame

class VariableEqualizerBlock(processRect):
    """
    Equalizes the contrast in a specific region within the inputted frame
    
    Balance between fully equalized contrast and the un-altered frame can be
    varied by setting the 'alpha' and 'beta' inputs.
    """
    beta = Float(0., iotype="in")
    alpha = Float(1., iotype="in")
    def process(self, rect, frame):
        x,y,w,h = rect
        subimg = np.array(frame[y:y+h,x:x+w]) 
        subimg = self.beta*subimg + self.alpha*cv2.equalizeHist(subimg)   
        frame[y:y+h,x:x+w]  = subimg
        return frame
        

class frameSlices(Component):
    """
    Collect slices of inputted frame using rectangle specifications. 
    
    This component is typically used to grab regions of interest of an image for
    GUI display.
    """
    def __init__(self, channels = [0,1,2]):
        super(frameSlices,self).__init__()    
        self.add("frame_in", Array(iotype="in"))
        self.add("rects_in", Array(iotype="in"))
        self.add("slices", List([ np.array([0,0]) ],iotype="out"))
        self.add("combined", Array(iotype="out"))

        self.add("zero_mean", Float(0., iotype="out"))
        
        self.channels = channels
        
    def combine(self,left, right):
        """Stack images horizontally.
        """
        h = max(left.shape[0], right.shape[0])
        w = left.shape[1] + right.shape[1]
        hoff = left.shape[0]
        
        shape = list(left.shape)
        shape[0] = h
        shape[1] = w
        
        comb = np.zeros(tuple(shape),left.dtype)
        
        # left will be on left, aligned top, with right on right
        comb[:left.shape[0],:left.shape[1]] = left
        comb[:right.shape[0],left.shape[1]:] = right
        
        return comb            
        
    def execute(self):
        comb = 150*np.ones((2,2))
        if self.rects_in.size > 0:
            self.slices = []
            for x,y,w,h in self.rects_in:
                output = self.frame_in[y:y+h,x:x+w]
                self.slices.append(output)
                
                comb = self.combine(output, comb)
        self.combined = comb
        self.zero_mean = self.slices[0].mean()
        

        