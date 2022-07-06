#!/usr/bin/python3
import PySimpleGUI as gui
import argparse
import sys
import vlc

# setup the loading window
gui.theme("DarkGrey11")
layout = [
    [gui.Text("Initializing")],
    [gui.ProgressBar(5, orientation="h", size=(20, 20))],
    ]
window = gui.Window("OptiClass GUI", layout, finalize=True)

# ignore linting for nvidia libs because dev environment won't always have all the modules
import jetson.inference # type: ignore
import jetson.utils # type: ignore
window[0].update(1)

# parse the command line
parser = argparse.ArgumentParser(description="Classify objects in a video feed and get information about them", formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="/dev/video0", nargs='?', help="URI of the input stream (often /dev/video#, or a local file)")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

try:
        opt = parser.parse_known_args()[0]
except:
        print("")
        parser.print_help()
        sys.exit(0)

window[0].update(2)

# load the specified network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)
window[0].update(3)

# create the gstreamer video source and output
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput("rtp://127.0.0.1:1234")
window[0].update(4)

# create the vlc instance and player
inst = vlc.Instance()
list_player = inst.media_list_player_new()
media_list = inst.media_list_new([])
list_player.set_media_list(media_list)
player = list_player.get_media_player()
player.set_xwindow(window['-WEBCAM-'].Widget.winfo_id())
media_list.add_media("rtp://127.0.0.1:1234")
list_player.set_media_list(media_list)
window[0].update(5)

window.close()

# setup the main window
layout = [
    [gui.Text("OptiClass")],
    [gui.Image('', size=(300, 170), key='-WEBCAM-')],
]

window['-WEBCAM-'].expand(True, True) # resize video window to fill space

while True:
	# capture the next image
    img = input.Capture()

	# detect objects in the image (with overlay)
    detections = net.Detect(img, overlay=opt.overlay)
    output.Render(img)