Getting started on Windows with Python(x,y)
-----------------------

- First, install Python(x,y) for windows, which you can download for free at:
 https://code.google.com/p/pythonxy/wiki/Downloads
Python(x,y) installs the python programming language on windows, 
plus a ton of additional scientific computing libraries and tools to go along with it. 
When you run the installer, there will be a panel where you can choose to install or not 
install different software libraries that you'd like/don't like. You have to go through 
and uncheck the one labelled "ETS" (it causes a bug). Then scroll through and check the one labelled "OpenCV".

- Next, go to:
http://openmdao.org/downloads-2/recent/
and download the latest release file (should look like "go-openmdao.py"). Download this
file anywhere you'd like, then run the file using python. The best way to do this is to
open a windows command prompt in the same directory as the "go-openmdao.py" file, then
run it by typing:

  `python go-openmdao.py`

  in the prompt, and pressing enter. It'll build a bunch of things into a folder in the same directory that will be named something like "openmdao-0.6.2".

- Then, download the webcam pulse detector code from https://github.com/thearn/webcam-pulse-detector/archive/master.zip

Now the easiest way to run/inspect/tweak the code is actually to use one of the programs that was installed with python(x,y), 
it's called 'Spyder'. It's a code editor for python that comes with some really neat programming features, and works very similar to MATLAB.

Open up Spyder, then make one tweak to the configuration settings: From the top menu, select:
Tools -> Preferences

From the list on the left, select console. Under the "Advanced settings" tab on the right, there's an option to change the path 
location of the python.exe file used by Spyder. Go ahead and change that to the "openmdao-0.6.2\Scripts\python.exe" that was made in the earlier step.

Kind of a lot of steps, but that should actually be it. When you extract the zip file that has my code, there will be a file called "get_pulse.py". 
Go ahead and load that into Spyder, and hit "run" (green button in the upper left). You might need to set the directory with my source code as the current working directory (top-right of the screen).

Troubleshooting
------------------

There's a chance you might run into a bug that I came across on windows where the data plot will look garbled/scratched. I think that sometimes python(x,y) 
installs an older version of OpenCV. To fix it, you just need to replace the file "cv2.pyd" in your c:/Python27/Lib/site-packages directory with one from https://dl.dropboxusercontent.com/u/1886126/cv2.pyd
(delete or over-write the existing one). Might not be needed for your machine though.
