from openmdao.lib.datatypes.api import Float, Dict, Array, List, Int, Bool
from openmdao.main.api import Component, Assembly
import numpy as np
import time
     
"""
Some 1D signal processing methods used for the analysis of image frames
"""
        
class PhaseController(Component):
    """
    Outputs either a convex combination of two floats generated from an inputted 
    phase angle, or a set of two default values
    
    The inputted phase should be an angle ranging from 0 to 2*pi
    
    The behavior is toggled by the boolean "state" input, which may be connected
    by another component or set directly by the user during a run
    
    (In short, this component can help make parts of an image frame flash in 
    sync to a detected heartbeat signal, in real time)
    """
    phase = Float(iotype="in")
    state = Bool(iotype="in")
    
    alpha = Float(iotype="out")
    beta = Float(iotype="out")
    
    def __init__(self, default_a, default_b,state = False):
        super(PhaseController,self).__init__()
        self.state = state
        self.default_a = default_a
        self.default_b = default_b
    
    def toggle(self):
        if self.state:
            self.state = False
        else:
            self.state = True
        return self.state
        
    def on(self):
        if not self.state:
            self.toggle()
    
    def off(self):
        if self.state:
            self.toggle()
    
    def execute(self):
        if self.state:
            t = (np.sin(self.phase) + 1.)/2.
            t = 0.9*t + 0.1
            self.alpha = t
            self.beta = 1-t
        else:
            self.alpha = self.default_a
            self.beta = self.default_b


class BufferFFT(Component):
    """
    Collects data from a connected input float over each run and buffers it
    internally into lists of maximum size 'n'.
    
    (So, each run increases the size of these buffers by 1.)
    
    Computes an FFT of this buffered data, along with timestamps and 
    correspondonding frequencies (hz), as output arrays.
    
    When the internal buffer is full to size 'n', the boolean 'ready' is 
    toggled to True. This indicates that this component is providing output 
    data corresponding to an n-point FFT. The 'ready' state can be outputed as
    a digital control to another component taking a boolean input.
    
    Can be reset to clear out internal buffers using the reset() method. This
    toggles the 'ready' state to False.
    """
    ready = Bool(False, iotype="out")
    fps = Float(iotype = "out")
    size = Int(iotype = "out")
    n = Int(iotype = "out")
    def __init__(self, n = 322, spike_limit = 5.):
        super(BufferFFT,self).__init__()
        self.n = n
        self.add("data_in", Float(iotype="in"))
        self.samples = []
        self.fps = 1.
        self.add("times", List(iotype="out"))
        self.add("fft", Array(iotype="out"))
        self.add("freqs", Array(iotype="out"))
        self.interpolated = np.zeros(2)
        self.even_times = np.zeros(2)
        
        self.spike_limit = spike_limit


    def get_fft(self):
        n = len(self.times)
        self.fps = float(n) / (self.times[-1] - self.times[0])
        self.even_times = np.linspace(self.times[0], self.times[-1], n)
        interpolated = np.interp(self.even_times, self.times, self.samples)
        interpolated = np.hamming(n) * interpolated
        self.interpolated = interpolated
        interpolated = interpolated - np.mean(interpolated)
        # Perform the FFT
        fft = np.fft.rfft(interpolated)
        self.freqs = float(self.fps)/n*np.arange(n/2 + 1)
        return fft      
    
    def find_offset(self):
        N = len(self.samples)
        for i in xrange(2,N):
            samples = self.samples[i:]
            delta =  max(samples)-min(samples)
            if delta < self.spike_limit:
                return N-i
    
    def reset(self):
        N = self.find_offset()
        self.ready = False
        self.times = self.times[N:]
        self.samples = self.samples[N:]

    def execute(self):
        self.samples.append(self.data_in)
        self.times.append(time.time())
        self.size = len(self.samples)
        if self.size > self.n:
            self.ready = True
            self.samples = self.samples[-self.n:]
            self.times = self.times[-self.n:]
        if self.size>4:
            self.fft = self.get_fft()
            if self.spike_limit:
                if max(self.samples)-min(self.samples) > self.spike_limit:
                    self.reset()

class bandProcess(Component):
    """
    Component to isolate specific frequency bands
    """
    hz = Float(iotype="out")
    peak_hz = Float(iotype="out")
    phase = Float(iotype="out")
    def __init__(self, limits = [0.,3.], make_filtered = True, 
                 operation = "pass"):
        super(bandProcess,self).__init__()
        self.add("freqs_in",Array(iotype="in"))
        self.add("fft_in", Array(iotype="in"))
        
        self.add("freqs", Array(iotype="out"))
        self.make_filtered = make_filtered
        if make_filtered:
            self.add("filtered", Array(iotype="out"))
        self.add("fft", Array(iotype="out"))
        self.limits = limits
        self.operation = operation
        
    def execute(self):
        if self.operation == "pass":
            idx = np.where((self.freqs_in > self.limits[0]) 
                           & (self.freqs_in < self.limits[1]))
        else:
            idx = np.where((self.freqs_in < self.limits[0]) 
                           & (self.freqs_in > self.limits[1]))
        self.freqs = self.freqs_in[idx] 
        self.fft = np.abs(self.fft_in[idx])**2
        
        if self.make_filtered:
            fft_out = 0*self.fft_in
            fft_out[idx] = self.fft_in[idx]
            
            if len(fft_out) > 2:
                self.filtered = np.fft.irfft(fft_out) 
                
                self.filtered = self.filtered / np.hamming(len(self.filtered))
        try:
            maxidx = np.argmax(self.fft)
            self.peak_hz = self.freqs[maxidx]
            self.phase = np.angle(self.fft_in)[idx][maxidx]
        except ValueError:
            pass #temporary fix for no-data situations

class Cardiac(bandProcess):
    """
    Component to isolate portions of a pre-computed time series FFT 
    corresponding to human heartbeats
    """
    
    def __init__(self, bpm_limits = [50,160]):
        super(Cardiac,self).__init__()
        self.add("bpm", Float(iotype="out"))
        self.limits = [bpm_limits[0]/60., bpm_limits[1]/60.]
        
    def execute(self):
        super(Cardiac,self).execute()
        self.freqs = 60*self.freqs
        self.bpm = 60*self.peak_hz
        
