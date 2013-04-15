from openmdao.lib.datatypes.api import Float, Dict, Array, List, Int
from openmdao.main.api import Component, Assembly

from imageProcess import RGBSplit, RGBmuxer, equalizeContrast, Grayscale
from detectors import faceDetector
from sliceops import frameSlices, equalizeBlock, drawRectangles
from signalProcess import BufferFFT, Cardiac, PhaseController
from numpy import mean
import time, cv2

class findFacesAndPulses(Assembly):
    """
    Detects human faces, then isolates foreheads
    
    Collects the mean value of the green channel in the forehead locations 
    over time
    
    Uses this information to estimate the detected individual's heartbeat
    """
    def __init__(self):
        super(findFacesAndPulses, self).__init__()
        
        #-----------assembly-level I/O-----------
        
        #input array
        self.add("frame_in", Array(iotype="in"))
        #output array
        self.add("frame_out", Array(iotype="out"))
        #array of detected faces (as single frame)
        self.add("faces", Array(iotype="out"))
        
        #-----------components-----------
        
        #splits input color image into R,G,B channels
        self.add("splitter", RGBSplit())
        self.driver.workflow.add("splitter")
        
        #converts input color image to grayscale
        self.add("gray", Grayscale())
        self.driver.workflow.add("gray")        
        
        #equalizes contast on the grayscale'd input image
        self.add("eq", equalizeContrast())
        self.driver.workflow.add("eq")       
        
        #finds faces within the grayscale's and contast-adjusted input image
        self.add("find_faces", faceDetector())
        self.driver.workflow.add("find_faces")
        
        #collects subimage samples of the detected faces
        self.add("grab_faces", frameSlices())
        self.driver.workflow.add("grab_faces")
        
        #collects subimage samples of the detected foreheads
        self.add("grab_foreheads", frameSlices())
        self.driver.workflow.add("grab_foreheads")     
        
        #highlights the locations of detected faces using contrast equalization
        self.add("highlight_faces", equalizeBlock(channels=[0,1,2]))
        self.driver.workflow.add("highlight_faces")
        
        #highlights the locations of detected foreheads using 
        #contrast equalization (green channel only)
        self.add("highlight_fhd", equalizeBlock(channels=[1], zerochannels=[0,2]))
        self.driver.workflow.add("highlight_fhd")
        
        #collects data over time to compute a 1d temporal FFT
        self.add("fft", BufferFFT(quality_limit = 13.))
        self.driver.workflow.add("fft")
        
        #takes in a computed FFT and estimates cardiac data
        self.add("heart", Cardiac(bpm_limits = [50,160]))
        self.driver.workflow.add("heart")
        
        #toggles flashing of the detected foreheads in sync with the detected 
        #heartbeat
        self.add("bpm_flasher", PhaseController(default_a=1., default_b=0.,
                                                state = True))
        self.driver.workflow.add("bpm_flasher")   
        
        
        #-----------connections-----------
        
        #pass image frames to RGB splitter & grayscale converters
        self.connect("frame_in", "splitter.frame_in")
        self.connect("frame_in", "gray.frame_in")
        
        #pass grayscaled image to contrast equalizer
        self.connect("gray.frame_out", "eq.frame_in")
        
        #contrast adjusted grayscale image to face detector
        self.connect("eq.frame_out", "find_faces.frame_in")
        
        #pass original image frame and detected faces locations 
        #to the face highlighter
        self.connect("frame_in", "highlight_faces.frame_in")
        self.connect("find_faces.detected", "highlight_faces.rects_in")
        
        #pass original image frame and detected face locations
        #to the forehead highlighter
        self.connect("highlight_faces.frame_out", "highlight_fhd.frame_in")
        self.connect("find_faces.foreheads", "highlight_fhd.rects_in")
        
        #pass original image frame and detected face locations
        #to the face subimage collector
        self.connect("find_faces.detected", "grab_faces.rects_in")
        self.connect("eq.frame_out", "grab_faces.frame_in")
        
        
        #pass the green channel of the original image frame and detected 
        #face locations to the forehead subimage collector
        self.connect("find_faces.foreheads", "grab_foreheads.rects_in")
        self.connect("splitter.G", "grab_foreheads.frame_in")   
        
        #send the mean of the first detected forehead subimage (green channel)
        #to the buffering FFT component
        self.connect("grab_foreheads.slices[0].mean()", "fft.data_in")
        
        #Send the FFT outputs (the fft & associated freqs in hz) to the cardiac
        #data estimator
        self.connect("fft.fft", "heart.fft_in")
        self.connect("fft.freqs", "heart.freqs_in")
        
        #connect the estimated heartbeat phase to the forehead flashing controller
        self.connect("heart.phase", "bpm_flasher.phase")
        self.connect("fft.ready", "bpm_flasher.state")
        
        #connect the flash controller to the forehead highlighter 
        self.connect("bpm_flasher.alpha", "highlight_fhd.alpha")
        self.connect("bpm_flasher.beta", "highlight_fhd.beta")
        
        #connect collection of all detected faces up to assembly level for output
        self.connect("grab_faces.combined", "faces")
        
        #output the frame containing the forehead highlighting 
        #up to assembly level, as the final output frame.
        self.connect("highlight_fhd.frame_out", "frame_out") 
        
        
        
class getFPS(Component):
    """
    Grabs an image frame, computes FPS, then overlays it.
    """
    fps = Float(iotype="out")
    def __init__(self, n = 100):
        super(getFPS, self).__init__()
        self.add("frame_in", Array(iotype="in"))
        self.add("frame_out", Array(iotype="out"))
        self.times = []
        self.n = n
        
    def execute(self):
        self.times.append(time.time())
        N = len(self.times)
        if N>2:
            self.fps = float(N) / (self.times[-1] - self.times[0])
        
        if N > self.n:
            self.times = self.times[-self.n:]
            
        cv2.putText(self.frame_in,"%0.0f %s" % (self.fps,"fps"),(100,100),
                    cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
        self.frame_out = self.frame_in
        