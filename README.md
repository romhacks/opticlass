# Opticlass: Searching Wikipedia So You Don't Have To

Opticlass is a graphical application that uses a given imagenet network to classify a video feed, and provides the user with a description of the object by scraping Wikipedia.

![image](https://user-images.githubusercontent.com/42524580/178066833-c09ada50-c98b-49ef-b0f9-4c38f0ba4097.png)


This project expects a[n](https://linustechtips.com/topic/22552-is-it-a-nvidia-insert-noun-or-an-nvidia-insert-noun/) NVIDIA Jetson environment and the jetson-inference library and project to be installed.

## Algorithm

We are able to use a pretrained model on imagenet (by default `googlenet`) to classify the image, which then is used to obtain a summary of a Wikipedia article with the same name. The `wikipedia` library is able to automatically follow redirects and the like, which is useful in getting the most results possible. We only run an inference every 10 frames to preserve resources, however this rate is boosted when a change is detected to make the UI feel more responsive.

## Setup
First, install requirements:

`pip3 install -r requirements.txt`

Then you can run the program, replacing `/dev/video0` with your video source:

`python3 ./opticlass.py /dev/video0`

Optionally include `--network {name}` to use a custom network. This isn't heavily tested.

If you have issues getting the window to show, run `export DISPLAY=:0` prior to the python command.

## Video

A video demonstration of the setup and program can be seen [here](https://example.com).
