#!/usr/bin/env python3

import PySimpleGUI as gui
import argparse
import sys
from time import sleep
from PIL import Image
import io
import numpy as np

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
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

window[0].update(2)

# load the recognition network
net = jetson.inference.imageNet("googlenet")

# create video sources
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
window[0].update(3)

# capture the first frame to get the camera pipeline going
img = input.Capture()
window[0].update(4)

window.close()

# setup the main window
layout = [
    [gui.Text("OptiClass", key = "-DESC-")],
    [gui.Image('', size=(300, 170), key="-WEBCAM-")],
]

window = gui.Window("OptiClass GUI", layout, finalize=True)
#window['-WEBCAM-'].expand(True, True) # resize video window to fill space

# process frames until the user exits
while True:
	event, values = window.read(timeout=0)
	print(event, values)

	# stop evaluating if window closed
	if event == gui.WIN_CLOSED:
		break

	# capture the next image
	img = input.Capture()
	window.refresh()

	#classify
	class_idx, confidence = net.Classify(img)
	class_desc = net.GetClassDesc(class_idx)

	# print the detections
	print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))

	img = Image.fromarray(jetson.utils.cudaToNumpy(img))
	img.thumbnail((400,400))
	bio = io.BytesIO()
	img.save(bio, format="PNG")
	window["-WEBCAM-"].update(data=bio.getvalue())
	window["-DESC-"].update(class_desc)

	# exit on input/output EOS
	if not input.IsStreaming():
		break
