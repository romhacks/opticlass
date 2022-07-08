#!/usr/bin/env python3

import PySimpleGUI as gui
import webbrowser
import argparse
import sys
from PIL import Image
import io
import threading
import wikipedia
import textwrap as tr

from numpy import append

# setup the loading window
gui.theme("DarkGrey11")
layout = [
    [gui.Text("Initializing")],
    [gui.ProgressBar(4, orientation="h", size=(20, 20))],
]
window = gui.Window("OptiClass GUI", layout, finalize=True)

import jetson.inference  # type: ignore
import jetson.utils  # type: ignore

window[0].update(1)

# parse the command line
parser = argparse.ArgumentParser(
    description="Provide context for images determined from a video stream",
    formatter_class=argparse.RawTextHelpFormatter,
)

parser.add_argument(
    "input_URI", type=str, default="/dev/video0", nargs="?", help="URI of the input stream"
)
parser.add_argument(
    "--network",
    type=str,
    default="googlenet",
    help="pre-trained model to load (see below for options)",
)

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

# function for getting wikipedia definitions
def get_definition(word):
    try:
        definition = wikipedia.summary(word)
        return definition
    except:
        return None


def update_definition(word):  # this doesn't need to exist but I don't want to change it
    global definition
    definition = get_definition(word)


# setup the main window
classOut = [
    [gui.Text("Image is recognized as:")],
    [gui.Text("", key="-DESC-")],
    [gui.Button("Google it", key="-GOOGLE-"), gui.Button("Wikipedia", key="-WIKI-")],
    [gui.Button("Pause/Resume viewer", key="-TOGGLE-")],
]

defOut = [
    [gui.Text("Definition:")],
    [gui.Text("", key="-DEF-")],
]

layout = [
    [
        gui.Image("", size=(250, 250), key="-WEBCAM-"),
        gui.Column(classOut),
    ],
    [gui.Column(defOut)],
]

window = gui.Window("OptiClass GUI", layout, finalize=True)


# main event loop
i = 0
past = 0
pastDisp = ""
threshold = 10
definition = ""
thread = threading.Thread()
running = True
while True:
    event, values = window.read(timeout=0)

    # stop evaluating if window closed
    if event == gui.WIN_CLOSED:
        break
    if event == "-GOOGLE-":  # if the user clicked the button, open a browser search
        webbrowser.open("https://google.com/search?q=" + pastDisp)
    if event == "-WIKI-":  # if the user clicked the button, open a wikipedia page
        webbrowser.open("https://en.wikipedia.org/wiki/" + pastDisp)
    if event == "-TOGGLE-":  # if the user clicked the button, toggle the input stream
        running = not running
    if running:
        # capture the next image
        img = input.Capture()
        window.refresh()

        if i == threshold:  # save cycles by only running this every 10 frames
            # classify the captured image
            class_idx, confidence = net.Classify(img)
            class_desc = net.GetClassDesc(class_idx).partition(",")[
                0
            ]  # we only want the first name of the class

            # print the detections
            print(
                "image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(
                    class_desc, class_idx, confidence * 100
                )
            )

            if (
                past == class_idx
            ):  # rudimentary smoothing by requiring two consecutive detections to be the same
                window["-DESC-"].update(class_desc)
                if (
                    class_desc != pastDisp and not thread.is_alive()
                ):  # if the definition has changed and the there isn't a pending query
                    thread = threading.Thread(
                        target=update_definition, args=(class_desc,)
                    )
                    thread.start()
                pastDisp = class_desc
                threshold = 10
            else:
                threshold = 2  # boost inference speed if the class changes
            past = class_idx
            i = 0
        img = Image.fromarray(
            jetson.utils.cudaToNumpy(img)
        )  # convert cudaimage to array then to PIL image
        img.thumbnail((250, 250))  # resize image to fit in window
        bio = io.BytesIO()
        img.save(
            bio, format="PNG"
        )  # slow but seems to be the only way to make gui behave
        window["-WEBCAM-"].update(data=bio.getvalue())
        try:
            definition = tr.fill(definition, 120)
        except:
            pass
        window["-DEF-"].update(definition)

        # exit on input/output EOS
        if not input.IsStreaming():
            break
        i += 1
