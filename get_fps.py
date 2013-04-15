from lib.device import Camera
from lib.processors import getFPS
from lib.interface import imshow
import numpy as np      


if __name__ == "__main__":
    camera = Camera(camera=0)
    processor = getFPS()
    while True:
        frame = camera.get_frame()
        
        processor.frame_in = frame
        processor.run()
        output = processor.frame_out
        
        imshow("fps", output)
        
        