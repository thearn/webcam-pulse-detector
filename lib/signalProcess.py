from openmdao.lib.datatypes.api import Float, Dict, Array, List, Int, Bool
from openmdao.main.api import Component, Assembly
import numpy as np
from scipy import ndimage
import time

class arrayDemux(Component):
    """
    Simple 1d array-to-matrix mux'er
    """
    def __init__(self, n_arrays):
        super(arrayDemux, self).__init__()
        self.n_arrays = n_arrays
        for i in xrange(n_arrays):
            self.add('data_in_%s' % str(i), Array(iotype='in', copy=None))
        self.add('data_out', Array(iotype='out', copy=None))
    
    def execute(self):
        M = len(self.data_in_0)
        data_out = np.zeros((self.n_arrays, M))
        for i in xrange(self.n_arrays):
            data_out[i] = self.get('data_in_%s' % str(i))  
        setattr(self, 'data_out', data_out)      

class PCAcomponent(Component):
    """
    performs principal component analysis
    on an input matrix
    """
    def __init__(self, output_dim = None):
        super(PCAcomponent, self).__init__()
        self.add('data_in', Array(iotype='in', copy=None))
        self.add('data_out', Array(iotype='out', copy=None))
        self.output_dim = output_dim
        self.trained = False
            
    def train(self, data = None):
        if data == None:
            data = self.data_in
        if not self.output_dim:
            self.output_dim = data.shape[0]
        self.pcanode = PCANode(svd = True, 
                               output_dim = self.output_dim)
        self.pcanode.train(data.T)
        self.trained = True
        self.S = self.pcanode.d

    def execute(self):
        if not self.trained: 
            self.train()
        self.data_out = self.pcanode(self.data_in.T).T       
        
class PhaseController(Component):
    """
    Outputs either a convex combination generated from an inputted phase angle,
    or a set of two default values, based on the state of a toggleable boolean.
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

class bufferTemporal(Component):            
    ready = Bool(False, iotype="out")
    def __init__(self, n = 322, quality_limit = 12.):
        super(bufferTemporal,self).__init__()
        self.n = n
        self.quality_limit = quality_limit
        self.add("data_in", Float(iotype="in"))
        self.add("samples", List(iotype="out"))
        self.add("times", List(iotype="out"))

    def find_offset(self):
        N = len(self.samples)
        for i in xrange(2,N):
            samples = self.samples[i:]
            delta =  max(samples)-min(samples)
            if delta < self.quality_limit:
                return N-i

    def reset(self):
        N = self.find_offset()
        self.ready = False
        self.times = self.times[N:]
        self.samples = self.samples[N:]

    def execute(self):
        self.samples.append(self.data_in)
        self.times.append(time.time())
        N = len(self.samples)
        if N > self.n:
            self.ready = True
            self.samples = self.samples[-self.n:]
            self.times = self.times[-self.n:]
            
        if self.quality_limit:
            if max(self.samples)-min(self.samples) > self.quality_limit:
                self.reset()

class FFT(Component):
    def __init__(self, n = 322, quality_limit = 5.):
        super(FFT,self).__init__()
        self.n = n
        self.add("samples", List(iotype="in"))
        self.fps = 1.
        self.add("times", List(iotype="in"))
        self.add("fft", Array(iotype="out"))
        self.add("freqs", Array(iotype="out"))
        self.interpolated = np.zeros(2)
        self.even_times = np.zeros(2)
        
    def get_fft(self):
        n = len(self.times)
        self.fps = float(n) / (self.times[-1] - self.times[0])
        self.even_times = np.linspace(self.times[0], self.times[-1], n)
        interpolated = np.interp(self.even_times, self.times, self.samples)
        interpolated = np.hamming(n) * interpolated
        self.interpolated = interpolated
        # Perform the FFT
        fft = np.fft.rfft(interpolated)
        self.freqs = float(self.fps)/n*np.arange(n/2 + 1)
        return fft  
    
    def execute(self):
        N = len(self.samples)
        if N>4:
            self.fft = self.get_fft()


class BufferFFT(Component):
    ready = Bool(False, iotype="out")
    def __init__(self, n = 322, quality_limit = 5.):
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
        
        self.quality_limit = quality_limit


    def get_fft(self):
        n = len(self.times)
        self.fps = float(n) / (self.times[-1] - self.times[0])
        self.even_times = np.linspace(self.times[0], self.times[-1], n)
        interpolated = np.interp(self.even_times, self.times, self.samples)
        interpolated = np.hamming(n) * interpolated
        self.interpolated = interpolated
        # Perform the FFT
        fft = np.fft.rfft(interpolated)
        self.freqs = float(self.fps)/n*np.arange(n/2 + 1)
        return fft      
    
    def find_offset(self):
        N = len(self.samples)
        for i in xrange(2,N):
            samples = self.samples[i:]
            delta =  max(samples)-min(samples)
            if delta < self.quality_limit:
                return N-i
    
    def reset(self):
        N = self.find_offset()
        self.ready = False
        self.times = self.times[N:]
        self.samples = self.samples[N:]

    def execute(self):
        self.samples.append(self.data_in)
        self.times.append(time.time())
        N = len(self.samples)
        if N > self.n:
            self.ready = True
            self.samples = self.samples[-self.n:]
            self.times = self.times[-self.n:]
        if N>4:
            self.fft = self.get_fft()
            if self.quality_limit:
                if max(self.samples)-min(self.samples) > self.quality_limit:
                    self.reset()

class Meyer(Component):
    power = Float(iotype="out")
    def __init__(self, limits = [0.09,.11]):
        super(Meyer,self).__init__()
        self.add("freqs_in",Array(iotype="in"))
        self.add("fft_in", Array(iotype="in"))
        
        self.limits = limits
        
    def execute(self):
        idx = np.where((self.freqs_in > self.limits[0]) 
                       & (self.freqs_in < self.limits[1]))
        freqs = self.freqs_in[idx] 
        fft = np.abs(self.fft_in[idx])
        
        print fft.sum()/np.abs(self.fft_in).sum()


class Cardiac(Component):
    bpm = Float(iotype="out")
    def __init__(self, bpm_limits = [50,160]):
        super(Cardiac,self).__init__()
        self.add("freqs_in",Array(iotype="in"))
        self.add("fft_in", Array(iotype="in"))
        
        self.add("freqs", Array(iotype="out"))
        self.add("filtered", Array(iotype="out"))
        self.add("fft", Array(iotype="out"))
        self.add("phase",Float(iotype="out"))
        self.bpm_limits = bpm_limits
        
    def execute(self):
        idx = np.where((self.freqs_in > self.bpm_limits[0]/60.) 
                       & (self.freqs_in < self.bpm_limits[1]/60.))
        self.freqs = 60*self.freqs_in[idx] 
        self.fft = np.abs(self.fft_in[idx])
        
        fft_out = 0*self.fft_in
        fft_out[idx] = self.fft_in[idx]
        
        if len(fft_out) > 2:
            self.filtered = np.fft.irfft(fft_out) 
            
            self.filtered = self.filtered / np.hamming(len(self.filtered))
            
        try:
            maxidx = np.argmax(self.fft)
            self.bpm = self.freqs[maxidx]
            self.phase = np.angle(self.fft_in)[idx][maxidx]
        except:
            pass
        
if __name__ == "__main__":
    proc = Kalman()
    while True:
        x=np.random.randn(10)[0]
        proc.data_in = x
        proc.run()
        
        print x,proc.filtered