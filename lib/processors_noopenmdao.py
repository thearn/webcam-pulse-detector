import numpy as np
import pandas as pd
import time
import cv2
import pylab
import os
import sys


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


default_color = (100, 255, 100)
red_color = (100, 100, 255)


class findFaceGetPulse(object):

    def __init__(self, fps=None, running_on_video=False):
        self.running_on_video = running_on_video
        self.fps = fps
        self.seconds_per_frame = 1 / fps if running_on_video else None

        self.fps_calculator_ticks = None
        self.fps_calculator_start = None
        # we need few seconds to calculate FPS
        self.fps_calculator_min_seconds = 3

        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.buffer_size = 250

        self.last_face_rects = pd.DataFrame(columns=['x', 'y', 'h', 'w'])
        self.fixed_face = None
        
        # restart tracking if face can't be detected for 0.5 seconds
        self.no_face_tolerance = 0.5
        self.no_face_counter = 0
        self.tracking_running = False

        # start calculating heart rate if face is stable for 2 seconds
        self.stable_face_threshold = 2

        self.tracking_batch_size = 50
        self.results = []
        self.bpm_buffer = []
        self.stable_face_counter = 0

        self.frame_i = None
        self.gray = None
        self.data_buffer = []
        self.times = []
        self.samples = []
        self.freqs = []
        self.fft = []
        self.slices = [[0]]
        self.t0 = None
        self.bpm = 0
        dpath = resource_path("haarcascade_frontalface_alt.xml")
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        self.face_cascade = cv2.CascadeClassifier(dpath)

        self.face_rect = None
        self.output_dim = 13

        self.idx = 1
        self.find_faces = True

    def find_faces_toggle(self):
        self.find_faces = not self.find_faces
        return self.find_faces

    def get_faces(self):
        return

    def draw_rect(self, rect, col=(0, 255, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), default_color, 1)

    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.face_rect
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]

    def get_subface_means(self, coord):
        x, y, w, h = coord
        subframe = self.frame_in[y:y + h, x:x + w, :]
        v1 = np.mean(subframe[:, :, 0])
        v2 = np.mean(subframe[:, :, 1])
        v3 = np.mean(subframe[:, :, 2])

        return (v1 + v2 + v3) / 3.

    def plot(self):
        data = np.array(self.data_buffer).T
        np.savetxt("data.dat", data)
        np.savetxt("times.dat", self.times)
        freqs = 60. * self.freqs
        idx = np.where((freqs > 50) & (freqs < 180))
        pylab.figure()
        n = data.shape[0]
        for k in xrange(n):
            pylab.subplot(n, 1, k + 1)
            pylab.plot(self.times, data[k])
        pylab.savefig("data.png")
        pylab.figure()
        for k in xrange(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(self.times, self.pcadata[k])
        pylab.savefig("data_pca.png")

        pylab.figure()
        for k in xrange(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(freqs[idx], self.fft[k][idx])
        pylab.savefig("data_fft.png")
        quit()

    def run(self, cam=None):
        if self.t0 is None:
            self.t0 = time.time()
            self.frame_i = 0
        else:
            self.frame_i += 1

        self.frame_out = self.frame_in
        self.bpm = 0

        # first we have to calculate fps
        # if it's not ready - skip everything else
        if not self.calculate_fps():
            cv2.putText(
                self.frame_out, "Calculating FPS",
                (10, 450), cv2.FONT_HERSHEY_PLAIN, 1.5, red_color, 2)
            return

        # try to detect face
        self.detect_face()

        if self.face_rect is None:
            self.no_face_counter += 1

            # too long without a face, restart tracking
            if self.no_face_counter > self.no_face_tolerance * self.fps:
                print('no face reset')
                self.clear_buffers()

            # otherwise - skip this frame but don't stop reset tracking just yet
        else:
            self.no_face_counter = 0

            # if face is out of range - clear buffers and stop tracking
            if self.current_face_out_of_range():
                #if self.stable_face_counter:
                print('out of range reset')
                self.clear_buffers()
            else:
                # we've got a stable face
                if not self.tracking_running:
                    self.stable_face_counter += 1

                    # check if face is stable long enough, start tracking if it's
                    if self.stable_face_counter >= self.stable_face_threshold * self.fps:
                        self.tracking_running = True
                        print('stabilized')

            # tracking is running
            if self.tracking_running:
                self.track_rate()

                if self.bpm:
                    self.bpm_buffer.append(self.bpm)
                #     print("(BPM estimate: %0.1f bpm)" % (self.bpm))

                # if we have enough data - store the mean bpm
                if len(self.bpm_buffer) >= self.tracking_batch_size:
                    new_mean = np.mean(self.bpm_buffer)
                    print("(BPM estimate: %0.1f bpm. fps: %d)" % (new_mean, self.fps))
                    self.results.append(new_mean)
                    self.bpm_buffer = []

            self.draw_face_rect()

        # print menu
        if not self.tracking_running:
            self.print_start_menu(cam)
        else:
            self.print_tracking_menu(cam)

    def print_start_menu(self, cam):
        if cam is not None:
            cv2.putText(
                self.frame_out, "Press 'C' to change camera (current: %s)" % str(
                    cam),
                (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.25, default_color)

        cv2.putText(self.frame_out, "Press 'Esc' to quit",
                   (10, 75), cv2.FONT_HERSHEY_PLAIN, 1.25, default_color)

    def print_tracking_menu(self, cam):
        if cam is not None:
            cv2.putText(
                self.frame_out, "Press 'C' to change camera (current: %s)" % str(
                    cam),
                (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.25, default_color)

        cv2.putText(
            self.frame_out, "Press 'S' to restart",
                   (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, default_color)
        cv2.putText(self.frame_out, "Press 'D' to toggle data plot",
                   (10, 75), cv2.FONT_HERSHEY_PLAIN, 1.5, default_color)
        cv2.putText(self.frame_out, "Press 'Esc' to quit",
                   (10, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, default_color)

    def clear_buffers(self):
        self.data_buffer, self.times = [], []
        self.last_face_rects = self.last_face_rects.iloc[0:0]
        self.bpm_buffer = []
        self.stable_face_counter = 0

    def detect_face(self):
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,
                                                  cv2.COLOR_BGR2GRAY))
        faces = []
        detected = list(self.face_cascade.detectMultiScale(self.gray,
                                                           scaleFactor=1.3,
                                                           minNeighbors=4,
                                                           minSize=(50, 50),
                                                           flags=cv2.CASCADE_SCALE_IMAGE))

        if len(detected) > 0:
            detected.sort(key=lambda a: a[-1] * a[-2])
            biggest_face = detected[-1]
            faces.append(biggest_face)
            self.face_rect = biggest_face
        else:
            # face not found
            self.face_rect = None
            return

    def draw_face_rect(self):
        forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
        self.draw_rect(self.face_rect, col=(255, 0, 0))
        x, y, w, h = self.face_rect
        cv2.putText(self.frame_out, "Face",
                    (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, default_color)
        self.draw_rect(forehead1)

    def face_dict_to_rect(selfs, face_dict):
        return (int(face_dict['x']),
                int(face_dict['y']),
                int(face_dict['w']),
                int(face_dict['h']))

    def is_face_close(self, face1, face2):
        delta = .07
        d_width = face1['w'] * delta
        d_height = face1['h'] * delta

        return abs(face1['x'] - face2['x']) < d_width and \
               abs(face1['w'] - face2['w']) < d_width and \
               abs(face1['y'] - face2['y']) < d_height and \
               abs(face1['h'] - face2['h']) < d_height

    def current_face_out_of_range(self):
        out_of_range = False
        
        # take current face rectangle
        x, y, w, h = self.face_rect
        face_dict = {'x': x, 'y': y, 'h': h, 'w': w}

        # append current face rectangle to a list
        self.last_face_rects = self.last_face_rects.append(face_dict, ignore_index=True)
        self.last_face_rects = self.last_face_rects.tail(20)

        # get average frame as a new candidate
        face_candidate = self.last_face_rects.mean()

        # if we don't have fixed frame, or current candidate is to far - reset
        if self.fixed_face is None or not self.is_face_close(face_candidate, self.fixed_face):
            out_of_range = True
            self.fixed_face = face_candidate
            
        self.face_rect = self.face_dict_to_rect(self.fixed_face)
        
        return out_of_range

    def track_rate(self):
        forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
        self.draw_rect(forehead1)

        x, y, w, h = forehead1
        avg_val = self.get_subface_means(forehead1) / (h * w)

        # using fps to calculate time for video
        # and real time when working with camera
        if self.running_on_video:
            self.times.append(self.frame_i * self.seconds_per_frame)
        else:
            self.times.append(time.time() - self.t0)

        self.data_buffer.append(avg_val)

        self.data_buffer[-1] = np.mean(self.data_buffer[-2:])

        L = len(self.data_buffer)

        # trim the data buffer and times buffer, we don't need more then buffer_size
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            L = self.buffer_size

        processed = np.array(self.data_buffer)
        self.samples = processed
        if L > 10:
            self.output_dim = processed.shape[0]

            even_times = np.linspace(self.times[0], self.times[-1], L)
            interpolated = np.interp(even_times, self.times, processed)
            interpolated = np.hamming(L) * interpolated
            interpolated = interpolated - np.mean(interpolated)
            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            self.fft = np.abs(raw)
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)

            freqs = 60. * self.freqs
            idx = np.where((freqs > 50) & (freqs < 180))

            pruned = self.fft[idx]
            phase = phase[idx]

            pfreq = freqs[idx]
            self.freqs = pfreq
            self.fft = pruned
            idx2 = np.argmax(pruned)

            t = (np.sin(phase[idx2]) + 1.) / 2.
            t = 0.9 * t + 0.1
            alpha = t
            beta = 1 - t

            bpm_estimate = self.freqs[idx2]
            self.idx += 1

            x, y, w, h = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
            r = alpha * self.frame_in[y:y + h, x:x + w, 0]
            g = alpha * \
                self.frame_in[y:y + h, x:x + w, 1] + \
                beta * self.gray[y:y + h, x:x + w]
            b = alpha * self.frame_in[y:y + h, x:x + w, 2]
            self.frame_out[y:y + h, x:x + w] = cv2.merge([r,
                                                          g,
                                                          b])
            x1, y1, w1, h1 = self.face_rect
            self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]
            gap = (self.buffer_size - L) / self.fps
            if gap:
                text = "(estimate: %0.1f bpm, wait %0.0f s)" % (bpm_estimate, gap)
            else:
                text = "(estimate: %0.1f bpm)" % (bpm_estimate)
                self.bpm = bpm_estimate

            tsize = 1
            cv2.putText(self.frame_out, text,
                       (int(x - w / 2), int(y)), cv2.FONT_HERSHEY_PLAIN, tsize, default_color)

    def calculate_fps(self):
        if self.fps:
            return True

        # if we didn't start counting - let's start
        if not self.fps_calculator_start:
            self.fps_calculator_start = time.time()
            self.fps_calculator_ticks = 0

            return

        # we started calculating the FPS
        self.fps_calculator_ticks += 1

        # time elapsed
        seconds = time.time() - self.fps_calculator_start

        if seconds >= self.fps_calculator_min_seconds:
            # calculate frames per second
            self.fps = self.fps_calculator_ticks / seconds

            return True
