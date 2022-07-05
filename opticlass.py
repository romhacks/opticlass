#!/usr/bin/env python3
# ignore linting for nvidia libs because dev environment won't always have all the modules
import jetson.inference # type: ignore
import jetson.utils # type: ignore
import PySimpleGUI as gui
import argparse
import sys

# parse the command line
parser = argparse.ArgumentParser(description="Classify objects in a video feed and get information about them", formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream (often /dev/video#, or a local file)")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

try:
        opt = parser.parse_known_args()[0]
except:
        print("")
        parser.print_help()
        sys.exit(0)

# setup the window
gui.theme("DarkGrey11")
layout = [
    [gui.Text("Initializing")],
    [gui.ProgressBar(2, orientation="h")],
    ]
window = gui.Window("OptiClass GUI", layout)

# load the specified network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)
window[0].update(1)

# create the video source
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
window[0].update(2)