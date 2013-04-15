from openmdao.lib.datatypes.api import Float, Dict, Array, List, Int, Bool
from openmdao.main.api import Component, Assembly
import numpy as np
from scipy import ndimage
import time

class RGB_sample_avg(Component):
    r = Float(0.,iotype="out")
    g = Float(0.,iotype="out")
    b = Float(0.,iotype="out")
    def __init__(self, n_arrays):
        super(RGB_sample_avg, self).__init__()
        self.add('frame_in', Array(iotype='in'))
        self.add('rects_in', Array(iotype='in'))
        
    def execute(self):
        if len(self.rects_in) > 0:
            x,y,w,h = self.rects_in[0]
            
            R = self.frame_in[x:x+w,y:y+h,0]
            G = self.frame_in[x:x+w,y:y+h,1]
            B = self.frame_in[x:x+w,y:y+h,2]
            
            self.r = R.mean()
            self.g = G.mean()
            self.b = B.mean()
        
        