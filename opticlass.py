#!/usr/bin/env python3

import PySimpleGUI as gui
import argparse
import sys
from PIL import Image
import io

# setup the loading window
gui.theme("DarkGrey11")
layout = [
    [gui.Text("Initializing")],
    [gui.ProgressBar(4, orientation="h", size=(20, 20))],
    ]
window = gui.Window("OptiClass GUI", layout, finalize=True)

import jetson.inference # type: ignore
import jetson.utils # type: ignore

window[0].update(1)

# parse the command line
parser = argparse.ArgumentParser(description="Provide context for images determined from a video stream", 
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

window[0].update(2)

# load the recognition network
net = jetson.inference.imageNet(opt.network)

# create video sources
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
window[0].update(3)

# capture the first frame to get the gstreamer pipeline working
img = input.Capture()
window[0].update(4)

window.close()

# setup the main window
layout = [
    [gui.Text("OptiClass", key = "-DESC-")],
    [gui.Image('', size=(300, 170), key="-WEBCAM-")],
]

window = gui.Window("OptiClass GUI", layout, finalize=True)

#TODO: do we need to do this?
#window['-WEBCAM-'].expand(True, True) # resize video window to fill space

# main event loop
while True:
	event, values = window.read(timeout=0)
	print(event, values)

	# stop evaluating if window closed
	if event == gui.WIN_CLOSED:
		break

	# capture the next image
	img = input.Capture()
	window.refresh()

	# classify the captured image
	class_idx, confidence = net.Classify(img)
	class_desc = net.GetClassDesc(class_idx).partition(",")[0] # we only want the first name of the class

	# print the detections
	print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))

	img = Image.fromarray(jetson.utils.cudaToNumpy(img)) # convert cudaimage to numpy array then to PIL image
	img.thumbnail((400,400)) # resize image to fit in window
	bio = io.BytesIO()
	img.save(bio, format="PNG") # slow as shit but seems to be the only way to make gui behave
	window["-WEBCAM-"].update(data=bio.getvalue())
	window["-DESC-"].update(class_desc)

	# exit on input/output EOS
	if not input.IsStreaming():
		break
