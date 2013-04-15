webcam-pulse-detector
=====================

An app that detects the heart-rate of an individual using their webcam.

Requirements:
---------------

-Python v2.7+
-OpenCV v2.4+ (with python cv2 library)
-OpenMDAO v0.5.5+

Quickstart:
------------
-In a windows command window or Unix/OSX terminal, activate the openMDAO virtual python environment
-Run get_pulse.py

How to use:
----------
-A window will open showing a stream from your computer's webcam
-The application will search for a human face within the camera frames, and attempt to isolate an area on the forehead
-When a forehead location has been isolated, the user should press "S" on their keyboard to lock the location, and remain as still as possible
-The application will then track the variations in this location over time to measure a heartbeat. To view a stream of this data, press "D".
-The data display shows three data traces, from top to bottom: 1) raw optical intensity, 2) extracted heartbeat signal, 3) Power spectral density, with local maxima indicated the heartrate (in beats per minute). 
