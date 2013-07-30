import numpy as np
import time, cv2, mdp, pylab

class findFaceGetPulse(object):
    
    def __init__(self, bpm_limits=[], data_spike_limit=250, 
                 face_detector_smoothness=10):
        
        self.frame_in = np.zeros((10,10))
        self.frame_out = np.zeros((10,10))
        self.fps = 0
        self.buffer_size = 300
        self.window = np.hamming(self.buffer_size)
        self.data_buffer = []
        self.times = []
        self.t0 = time.time()
        
        self.face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt.xml")
        
        self.face_rect = [1,1,2,2]
        self.last_center = np.array([0,0])
        self.last_wh = np.array([0,0])
        self.output_dim = 8
        self.trained = False
        
        self.idx = 1
        self.find_faces = True
        
    def find_faces_toggle(self):
        self.find_faces = not self.find_faces
        
    def get_faces(self):
        return
    
    def shift(self,detected):
        x,y,w,h = detected
        center =  np.array([x+0.5*w,y+0.5*h])
        shift = np.linalg.norm(center - self.last_center)
        
        self.last_center = center
        return shift
        
    def draw_rect(self, rect):
        x,y,w,h = rect
        cv2.rectangle(self.frame_out, (x, y), (x+w, y+h), (0,255,0), 3)
        
    def get_subface_coord(self,fh_x, fh_y, fh_w, fh_h):
        x,y,w,h = self.face_rect
        return [int(x + w * fh_x - (w*fh_w / 2.0)), 
                    int(y + h * fh_y - (h*fh_h / 2.0)), 
                    int(w * fh_w), 
                    int(h * fh_h)]
    
    def get_subface_means(self, coord):
        x, y, w, h = coord
        subframe = self.frame_in[y:y+h,x:x+w,:]
        v1 = np.mean(subframe[:,:,0])
        v2 = np.mean(subframe[:,:,1])
        
        #v1 = np.std(subframe[:,:,0])
        #v2 = np.std(subframe[:,:,1])
        
        #print v1,v2
        return v1, v2

    def plot(self):
        data = np.array(self.data_buffer).T
        freqs = 60.*self.freqs
        idx = np.where((freqs > 50) & (freqs < 180))  
        pylab.figure()
        for k in xrange(10):
            pylab.subplot(10,1,k+1)
            pylab.plot(self.times, data[k])
        pylab.savefig("data.png")
        pylab.figure()
        for k in xrange(self.output_dim):
            pylab.subplot(self.output_dim,1,k+1)
            pylab.plot(self.times, self.pcadata[k])
        pylab.savefig("data_pca.png")
        
        pylab.figure()
        for k in xrange(self.output_dim):
            pylab.subplot(self.output_dim,1,k+1)
            pylab.plot(freqs[idx], self.fft[k][idx])
        pylab.savefig("data_fft.png")
        
        quit()

    def run(self):
        self.times.append(time.time() - self.t0)
        self.frame_out = self.frame_in
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in, 
                                                  cv2.COLOR_BGR2GRAY))
        
        if self.find_faces:
            detected = list(self.face_cascade.detectMultiScale(self.gray, 
                                                  scaleFactor=1.3,
                                                  minNeighbors=4,
                                                  minSize=(50, 50), 
                                                  flags=cv2.CASCADE_SCALE_IMAGE))
        
            if len(detected) > 0:
                detected.sort(key=lambda a:a[-1]*a[-2])
                
                if self.shift(detected[-1]) > 10:
                    self.face_rect = detected[-1]
        
        if set(self.face_rect) == set([1,1,2,2]):
            return
        
        x, y, w, h = self.face_rect
        
        forehead1 = self.get_subface_coord(0.4, 0.18, 0.25, 0.15)
        forehead2 = self.get_subface_coord(0.6, 0.18, 0.25, 0.15)
        cheek1 = self.get_subface_coord(0.3, 0.58, 0.15, 0.2)
        cheek2 = self.get_subface_coord(0.7, 0.58, 0.15, 0.2)
        
        self.draw_rect(self.face_rect)
        self.draw_rect(forehead1)
        self.draw_rect(forehead2)
        self.draw_rect(cheek1)
        self.draw_rect(cheek2)
        
        vals = self.get_subface_means(forehead1) + \
            self.get_subface_means(forehead2) + \
            self.get_subface_means(cheek1) + \
            self.get_subface_means(cheek2) + self.get_subface_means(self.face_rect) 
        
        self.data_buffer.append(vals)
        
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            processed = np.array(self.data_buffer)
            
            inp = (processed  - np.mean(processed , 0))/np.std(processed, axis=0, ddof=0)
            #inp = processed
            
            if self.idx % self.buffer_size == 0:
                self.trained = False
                self.idx = 1
            if not self.trained:
                print "PCA Train...", self.fps
                #self.pca = mdp.nodes.PCANode(output_dim=self.output_dim)
                self.pca = mdp.Flow([mdp.nodes.PCANode(output_dim=self.output_dim), 
                                     mdp.nodes.CuBICANode()])
                self.pca.train(inp)
                self.trained = True
            self.pcadata = self.pca.execute(inp).T 
            pcadata = np.zeros(self.pcadata.shape)
            
            self.times = self.times[-self.buffer_size:]
            #print "ready"
            
            self.fps = float(self.buffer_size) / (self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1], self.buffer_size)
            for i in xrange(self.output_dim):
                interpolated = np.interp(even_times, self.times, self.pcadata[i] )
                interpolated = self.window * interpolated
                pcadata[i] = interpolated
            self.fft = np.abs(np.fft.rfft(self.pcadata, axis=1))
            self.freqs = float(self.fps)/self.buffer_size*np.arange(self.buffer_size/2 + 1)
            
            #pylab.show()
            #quit()
            #print np.abs(fft)
            #print 
            #print 60*self.freqs
            
            self.idx += 1
            
    
