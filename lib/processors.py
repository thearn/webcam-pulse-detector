from openmdao.lib.datatypes.api import Float, Dict, Array, List, Int
from openmdao.main.api import Component, Assembly

from imageProcess import RGBSplit, RGBmuxer, equalizeContrast, Grayscale, showBPMtext
from detectors import faceDetector
from sliceops import frameSlices, VariableEqualizerBlock, drawRectangles
from signalProcess import BufferFFT, Cardiac, PhaseController
from numpy import mean
import time, cv2


class findFaceGetPulse(Assembly):
    """
    An openMDAO assembly to detect a human face in an image frame, and then 
    isolate the forehead.
    
    Collects and buffers mean value of the green channel in the forehead locations 
    over time, with each run.
    
    This information is then used to estimate the detected individual's heartbeat
    
    Basic usage: 
    
    -Instance this assembly, then create a loop over frames collected
    from an imaging device. 
    -For each iteration of the loop, populate the assembly's 
    'frame_in' input array with the collected frame, then call the assembly's run()
    method to conduct all of the analysis. 
    -Finally, display annotated results
    from the output 'frame_out' array.
    
    """
    def __init__(self, 
                 bpm_limits = [50,160],
                 data_spike_limit = 13.,
                 face_detector_smoothness = 10):
        super(findFaceGetPulse, self).__init__()
        
        #-----------assembly-level I/O-----------
        
        #input array
        self.add("frame_in", Array(iotype="in"))
        #output array
        self.add("frame_out", Array(iotype="out"))
        #array of detected faces (as single frame)
        self.add("faces", Array(iotype="out"))
        
        #-----------components-----------
        # Each component we want to use must be added to the assembly, then also
        # added to the driver's workflow 
        
        #splits input color image into R,G,B channels
        self.add("RGBsplitter", RGBSplit())
        self.driver.workflow.add("RGBsplitter")
        
        #converts input color image to grayscale
        self.add("grayscale", Grayscale())
        self.driver.workflow.add("grayscale")        
        
        #equalizes contast on the grayscale'd input image
        self.add("contrast_eq", equalizeContrast())
        self.driver.workflow.add("contrast_eq")       
        
        #finds faces within the grayscale's and contast-adjusted input image
        #Sets smoothness parameter to help prevent 'jitteriness' in the face tracking
        self.add("find_faces", faceDetector(smooth = face_detector_smoothness))
        self.driver.workflow.add("find_faces")
        
        #collects subimage samples of the detected faces
        self.add("grab_faces", frameSlices())
        self.driver.workflow.add("grab_faces")
        
        #collects subimage samples of the detected foreheads
        self.add("grab_foreheads", frameSlices())
        self.driver.workflow.add("grab_foreheads")     
        
        #highlights the locations of detected faces using contrast equalization
        self.add("highlight_faces", VariableEqualizerBlock(channels=[0,1,2]))
        self.driver.workflow.add("highlight_faces")
        
        #highlights the locations of detected foreheads using 
        #contrast equalization (green channel only)
        self.add("highlight_fhd", VariableEqualizerBlock(channels=[1], 
                                                         zerochannels=[0,2]))
        self.driver.workflow.add("highlight_fhd")
        
        #collects data over time to compute a 1d temporal FFT
        # 'n' sets the internal buffer length (number of samples)
        # 'spike_limit' limits the size of acceptable spikes in the raw measured
        # data. When exceeeded due to poor data, the fft component's buffers 
        # are reset
        self.add("fft", BufferFFT(n=425,
                                  spike_limit = data_spike_limit))
        self.driver.workflow.add("fft")
        
        #takes in a computed FFT and estimates cardiac data
        # 'bpm_limits' sets the lower and upper limits (in bpm) for heartbeat
        # detection. 50 to 160 bpm is a pretty fair range here.
        self.add("measure_heart", Cardiac(bpm_limits = bpm_limits))
        self.driver.workflow.add("measure_heart")
        
        #toggles flashing of the detected foreheads in sync with the detected 
        #heartbeat. the 'default_a' and 'default_b' set the nominal contrast
        #correction that will occur when phase pulsing isn't enabled.
        #Pulsing is set by toggling the boolean variable 'state'.
        self.add("bpm_flasher", PhaseController(default_a=1., 
                                                default_b=0.,
                                                state = True))
        self.driver.workflow.add("bpm_flasher")   
        
        self.add("show_bpm_text", showBPMtext())
        self.driver.workflow.add("show_bpm_text")
        
        #-----------connections-----------
        # here is where we establish the relationships between the components 
        # that were added above.
        
        #--First, set up the connectivity for components that will do basic
        #--input, decomposition, and annotation of the inputted image frame
        
        # pass image frames from the assembly-level input arrays to the RGB 
        # splitter & grayscale converters (separately)
        self.connect("frame_in", "RGBsplitter.frame_in")
        self.connect("frame_in", "grayscale.frame_in")
        
        #pass grayscaled image to the contrast equalizer
        self.connect("grayscale.frame_out", "contrast_eq.frame_in")
        
        #pass the contrast adjusted grayscale image to the face detector
        self.connect("contrast_eq.frame_out", "find_faces.frame_in")
        
        # now pass our original image frame and the detected faces locations 
        # to the face highlighter
        self.connect("frame_in", "highlight_faces.frame_in")
        self.connect("find_faces.detected", "highlight_faces.rects_in")
        
        # pass the original image frame and detected face locations
        # to the forehead highlighter
        self.connect("highlight_faces.frame_out", "highlight_fhd.frame_in")
        self.connect("find_faces.foreheads", "highlight_fhd.rects_in")
        
        # pass the original image frame and detected face locations
        # to the face subimage collector
        self.connect("find_faces.detected", "grab_faces.rects_in")
        self.connect("contrast_eq.frame_out", "grab_faces.frame_in")
        
        # --Now we set the connectivity for the components that will do the 
        # --actual analysis
        
        #pass the green channel of the original image frame and detected 
        #face locations to the forehead subimage collector
        self.connect("find_faces.foreheads", "grab_foreheads.rects_in")
        self.connect("RGBsplitter.G", "grab_foreheads.frame_in")   
        
        #send the mean of the first detected forehead subimage (green channel)
        #to the buffering FFT component
        #Should probably be an intermediate component here, but that isn't 
        #actually necessary - we can do a connection between expressions in
        #addition to input/output variables.
        #self.connect("grab_foreheads.slices[0]", "fft.data_in")
        self.connect("grab_foreheads.zero_mean", "fft.data_in")
        
        #Send the FFT outputs (the fft & associated freqs in hz) to the cardiac
        #data estimator
        self.connect("fft.fft", "measure_heart.fft_in")
        self.connect("fft.freqs", "measure_heart.freqs_in")
        
        #connect the estimated heartbeat phase to the forehead flashing controller
        self.connect("measure_heart.phase", "bpm_flasher.phase")
        self.connect("fft.ready", "bpm_flasher.state")
        
        #connect the flash controller to the forehead highlighter 
        self.connect("bpm_flasher.alpha", "highlight_fhd.alpha")
        self.connect("bpm_flasher.beta", "highlight_fhd.beta")
        
        #connect collection of all detected faces up to assembly level for output
        self.connect("grab_faces.combined", "faces")
        
        # text display of estimated bpm
        self.connect("highlight_fhd.frame_out", "show_bpm_text.frame_in") 
        self.connect("measure_heart.bpm", "show_bpm_text.bpm")
        self.connect("find_faces.detected[0][0]", "show_bpm_text.x")
        self.connect("find_faces.detected[0][1]", "show_bpm_text.y")
        self.connect("fft.fps", "show_bpm_text.fps")
        self.connect("fft.size", "show_bpm_text.size")
        self.connect("fft.n", "show_bpm_text.n")
        
        self.connect("fft.ready", "show_bpm_text.ready")
        self.connect("show_bpm_text.frame_out", "frame_out") 
        