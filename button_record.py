from gpiozero import Button, LED
from picamera2 import Picamera2
import cv2
import time
import os
import re

# Global Flags
is_recording = False
is_paused = False

# Cameras Setup
width = 1280
height = 720
fps = 13
frame_time = 1.0 / fps

cam0 = Picamera2(0)
cam0.configure(cam0.create_video_configuration({'size': (width, height), 'format': 'RGB888'}))
cam0.start()
time.sleep(2)

cam1 = Picamera2(1)
cam1.configure(cam1.create_video_configuration({'size': (width, height), 'format': 'RGB888'}))
cam1.start()
time.sleep(2)

# global recording files
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out0 = None
out1 = None
frame0 = None
frame1 = None


def prepare_recording_filenames():
    global out0, out1
    folder = 'recordings'

    # Check if 'recordings' folder exists, if not create it
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")

    # Regular expression to match file names like rec{n}-cam0.avi
    pattern = re.compile(r"rec(\d+)-cam0\.avi")

    max_n = -1

    # Check all files in the folder and find the max n
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            n = int(match.group(1))
            if n > max_n:
                max_n = n

    new_n = max_n + 1

    # Prepare new file names
    file_cam0 = os.path.join(folder, f"rec{new_n}-cam0.avi")
    file_cam1 = os.path.join(folder, f"rec{new_n}-cam1.avi")

    # Create VideoWriter objects with the new file names
    out0 = cv2.VideoWriter(file_cam0, fourcc, fps, (width, height))
    out1 = cv2.VideoWriter(file_cam1, fourcc, fps, (width, height))

    print(f"Next recording filenames:\n{file_cam0}\n{file_cam1}")


def on_start_button_pressed():
    global is_recording
    global is_paused

    if is_recording:
        is_recording = False
        is_paused = False
        led.off()
    else:
        # create file name
        prepare_recording_filenames()

        is_recording = True
        is_paused = False
        led.on()

    print("Start Button Pressed")


def on_pause_button_pressed():
    global is_recording
    global is_paused

    if is_recording:
        if is_paused:
            is_paused = False
            led.on()
            print("resumed")

        else:
            is_paused = True
            led.blink()
            print("paused")
    else:
        pass

    print("Pause Button Pressed")


led = LED(27)

start_button = Button(6, pull_up=True, bounce_time=0.1)
start_button.when_pressed = on_start_button_pressed

pause_button = Button(5, pull_up=True, bounce_time=0.1)
pause_button.when_pressed = on_pause_button_pressed

frame_time = 1.0 / fps

while True:
    frame_start_time = time.time()
    # Capture frames from both cameras
    frame0 = cam0.capture_array()
    frame1 = cam1.capture_array()

    # Condition for recording
    if is_recording and not is_paused:
        if out0 is not None:
            out0.write(frame0)
        if out1 is not None:
            out1.write(frame1)

    # Calculate time taken to capture the frame
    frame_duration = time.time() - frame_start_time

    # Sleep to maintain the exact frame rate
    sleep_time = frame_time - frame_duration
    if sleep_time > 0:
        time.sleep(sleep_time)

# Stop the cameras and close windows
cam1.stop()
cam0.stop()
