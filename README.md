![Alt text](http://i.imgur.com/2ngZopS.jpg "Screenshot")

webcam-pulse-detector
-----------------------

-------------------

UPDATE: Stand-alone (no dependancy) precompiled application now available!
 - Download for Windows 7 and 8: [webcam-pulse-detector_win.zip](http://sourceforge.net/projects/webcampulsedetector/files/webcam-pulse-detector_win.zip/download) (42 Mb) 
 - Download for Mac OSX 10.6 (and later): [webcam-pulse-detector_mac.zip](http://sourceforge.net/projects/webcampulsedetector/files/webcam-pulse-detector_mac.zip/download) (21 Mb)
 - Debian/Ubuntu/Mint Linux: Coming very soon. For now, it is recommended that you run from source on the `no_openmdao` branch if you just want to test things out.

The application can be run by simply executing the binary contained in the zip file for your platform.
This code can also be run from source by [following the instructions below](#running-from-source).
 
-------------------

A python code that detects the heart-rate of an individual using a common webcam or network IP camera. 
Tested on OSX 10.7, 10.8, 10.9, Ubuntu 13.04 (Ringtail), and Windows 7 & 8.

Inspired by reviewing recent work on [Eulerian Video Magnification](http://people.csail.mit.edu/mrub/vidmag/), 
with motivation to implement something visually comparable (though not necessarily identical in formulation) to their
pulse detection examples using [Python](http://python.org/) and [OpenCV](http://opencv.org/) (see https://github.com/brycedrennan/eulerian-magnification for a 
more general take on the offline post-processing methodology). 
This goal is comparable to those of a few previous efforts in this area 
(such as https://github.com/mossblaser/HeartMonitor).

This code was developed at [NASA Glenn Research Center](http://www.nasa.gov/centers/glenn) in 
support of [OpenMDAO](http://openmdao.org/), under the Aeronautical Sciences Project in NASA's 
[Fundamental Aeronautics Program](http://www.aeronautics.nasa.gov/fap/), as well as the Crew State Monitoring Element 
of the Vehicle Systems Safety Technologies Project, in NASA’s 
[Aviation Safety Program](http://www.aeronautics.nasa.gov/programs_avsafe.htm).

How it works:
-----------------
This application uses [OpenCV](http://opencv.org/) to find the location of the user's face, then isolate the forehead region. Data is collected
from this location over time to estimate the user's heart rate. This is done by measuring average optical
intensity in the forehead location, in the subimage's green channel alone (a better color mixing ratio may exist, but the 
blue channel tends to be very noisy). Physiological data can be estimated this way via a combination of 
[photoplethysmology](http://en.wikipedia.org/wiki/Photoplethysmogram) and the optical absorption 
characteristics of (oxy-) haemoglobin (see http://www.opticsinfobase.org/oe/abstract.cfm?uri=oe-16-26-21434). 

With good lighting and minimal noise due to motion, a stable heartbeat should be 
isolated in about 15 seconds. Other physiological waveforms (such as 
[Mayer waves](http://en.wikipedia.org/wiki/Mayer_waves)) should also be visible in the raw data stream.

Once the user's heart rate has been estimated, real-time phase variation associated with this 
frequency is also computed. This allows for the heartbeat to be exaggerated in the post-process frame rendering, 
causing the highlighted forehead location to pulse in sync with the user's own heartbeat.

Support for detection on multiple simultaneous individuals in a single camera's 
image stream is definitely possible, but at the moment only the information from one face 
is extracted for analysis.

The overall dataflow/execution order for the real-time signal processing looks like:

![Alt text](http://i.imgur.com/xS7O8U3.png "Signal processing")

This signal processing design is implemented in the OpenMDAO assembly object defined in
[lib/processors.py](lib/processors.py).

The definition of each component block used can be found in the source 
files [lib/imageProcess.py](lib/imageProcess.py), [lib/signalProcess.py](lib/signalProcess.py), and 
[lib/sliceops.py](lib/sliceops.py). The `@bin` and `@bout` blocks in the above graph denote assembly-level input and 
output.


Running from source:
---------------

### Basic requirements: 

- [Python v2.7+](http://python.org/)
- Numpy and Scipy 
- Pyserial (for serial port output support)
- [OpenCV v2.4+](http://opencv.org/), with the cv2 python bindings
 
OpenCV is a powerful open-source computer vision library, with a convenient 
numpy-compatible interface in the cv2 bindings.

**If you want to run from source and modify UI or data output behavior, and make only minor changes 
to the signal processing, you can checkout and run the `no_openmdao` branch with no further dependancies.**
This branch implements a 'flattened' version of the master branch's OpenMDAO assembly, but as a plain python object.

Both the `no_openmdao` branch and the precompiled binary applications contain support for real time serial port and UDP output of the estimated heart rate.

However, if you would like to make significant or exploratory changes to the signal processing code (eg. multichannel support, PCA/ICA data factorizations, better filters, etc.), you should run the master
branch with OpenMDAO support and build on what is already there (see instructions below).

### Requirements for master branch (OpenMDAO support):

- [OpenMDAO v0.9+](http://openmdao.org/)

OpenMDAO is an open-source engineering framework that serves as a convenient 
environment to containerize the required real-time analysis, and 
allow for that analysis to be easily tweaked to specification and compared with alternative designs. 
Upon installation, OpenMDAO is bootstrapped into its own Python 
virtualenv, which must be activated before use (see the Quickstart section below). OpenMDAO requires python 2.6+, numpy, scipy, and matplotlib 
(see http://openmdao.org/docs/getting-started/requirements.html)

**Running Windows, completely new to Python, but still would like to hack on the `master` branch source code? Full instructions for getting started with all requirements needed to
run this code are available [here](win_pythonxy.md)**

Quickstart:
------------

### On branch `no_openmdao`:

- Simply run `get_pulse.py` in the top level directory.

### On branch `master`:

- Activate the openMDAO virtual python environment in a command or terminal window. On Linux and OSX, this is done by
running (note the period):

```
. OpenMDAO/bin/activate
```
Or on Windows:

```
OpenMDAO\Scripts\activate
```

- In the activated environment, navigate to the downloaded source directory, and run get_pulse.py to start the application

```
python get_pulse.py
```

- If there is an error, try running `test_webcam.py` in the same directory to check if your openCV installation and webcam can be made to work
with this application.

Usage notes:
----------
- When run, a window will open showing a stream from your computer's webcam
- When a forehead location has been correctly detected and isolated, the user should press "S" on their 
keyboard to lock this location, and remain as still as possible (the camera 
stream window must have focus for the click to register). This freezes the acquisition location in place. This lock can
be released by pressing "S" again.
- To view a stream of the measured data as it is gathered, press "D". To hide this display, press "D" again.
- The data display shows three data traces, from top to bottom: 
   1. (top) raw optical intensity
   2. (bottom) Power spectral density, with local maxima indicating the heartrate (in beats per minute). 
- With consistent lighting and minimal head motion, a stable heartbeat should be 
isolated in about 15 seconds. A count-down is shown in the image frame.

TODO:
------
- There have been some requests for a video demo
- Instead of processing using the green channel alone, it is likely that some fixed combination of the statistics of the
R,G,B channels could instead be optimal (though I was unable to find a simple combination that was better than green
alone). If so, the mixing ratios might be determinable from the forward projection matrices of PCA or ICA operators 
computed on a set of mean value R,G, and B data gathered over a trial data set (and verified with different individuals 
under different lighting conditions).
- Support for multiple individuals
- Smoother tracking of data from foreheads, perhaps by buffering and registering or inverse-transforming image subframes

