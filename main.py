from gpiozero import Button, LED
from picamera2 import Picamera2
import cv2
import time
import os
import re
import numpy as np
from flask import Flask, Response, jsonify
from threading import Thread


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



frame0 = None
frame1 = None
is_recording = False
is_paused = False
out = None
fourcc = cv2.VideoWriter_fourcc(*'XVID')


def prepare_recording_filename():
    global out, fourcc
    folder = 'recordings'

    # Check if 'recordings' folder exists, if not create it
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")

    # Regular expression to match file names like rec{n}.avi
    pattern = re.compile(r"rec(\d+)\.avi")

    max_n = -1

    # Check all files in the folder and find the max n
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            n = int(match.group(1))
            if n > max_n:
                max_n = n

    new_n = max_n + 1

    # Prepare new file name
    file_path = os.path.join(folder, f"rec{new_n}.avi")

    # Create VideoWriter object with the new file name
    out = cv2.VideoWriter(file_path, fourcc , fps, (width * 2 + 5, height))

    print(f"Next recording filename:\n{file_path}")


def on_start_button_pressed():
    global led, is_recording, is_paused
    if is_recording:
        is_recording = False
        is_paused = False
        led.off()
    else:
        # create file name
        prepare_recording_filename()

        is_recording = True
        is_paused = False
        led.on()

    print("Start Button Pressed")


def on_pause_button_pressed():
    global led, is_recording, is_paused

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

# Flask Server Part

app = Flask(__name__)


def generate_frames_0():
    global frame0
    last_frame = None

    while True:
        if frame0 is not None and frame0 is not last_frame:
            _, buffer = cv2.imencode('.jpg', frame0)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            last_frame = frame0  # Update last_frame reference

        else:
            time.sleep(frame_time / 5)  # Avoid tight CPU-hogging loop


def generate_frames_1():
    global frame1
    last_frame = None

    while True:
        if frame1 is not None and frame1 is not last_frame:
            _, buffer = cv2.imencode('.jpg', frame1)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            last_frame = frame1  # Update last_frame reference

        else:
            time.sleep(frame_time / 5)  # Avoid tight CPU-hogging loop


@app.route('/video_feed_0')
def video_feed_0():
    return Response(generate_frames_0(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_1')
def video_feed_1():
    return Response(generate_frames_1(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Only for testing. Will be removed later
@app.route('/')
def index():
    return "<html><body><img src='/video_feed_0' width='640'><img src='/video_feed_1' width='640'></body></html>"


@app.route('/record', methods=['GET'])
def start_recording():
    global is_recording, is_paused, led
    if not is_recording:
        prepare_recording_filename()

        is_recording = True
        is_paused = False
        led.on()

    device_state = {
        "is_recording": is_recording,
        "is_paused": is_paused
    }
    return jsonify(device_state)


@app.route('/stop', methods=['GET'])
def stop_recording():
    global is_recording, is_paused, led

    if is_recording:
        is_recording = False
        is_paused = False
        led.off()

    device_state = {
        "is_recording": is_recording,
        "is_paused": is_paused
    }
    return jsonify(device_state)


@app.route('/pause', methods=['GET'])
def pause_recording():
    global is_recording, is_paused, led

    if is_recording:
        if not is_paused:
            is_paused = True
            led.blink()
            print("paused")

    device_state = {
        "is_recording": is_recording,
        "is_paused": is_paused
    }
    return jsonify(device_state)


@app.route('/resume', methods=['GET'])
def resume_recording():
    global is_recording, is_paused, led

    if is_recording:
        if is_paused:
            is_paused = False
            led.on()
            print("resumed")

    device_state = {
        "is_recording": is_recording,
        "is_paused": is_paused
    }
    return jsonify(device_state)


@app.route('/state', methods=['GET'])
def get_device_state():
    global is_recording, is_paused, led

    device_state = {
        "is_recording": is_recording,
        "is_paused": is_paused  # or True
    }
    return jsonify(device_state)


def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)


flask_thread = Thread(target=run_flask)
flask_thread.start()

while True:
    frame_start_time = time.time()
    # Capture frames from both cameras
    frame0 = cam0.capture_array()
    frame1 = cam1.capture_array()

    # Condition for recording
    if is_recording and not is_paused:
        if out is not None:
            # Create a 5px vertical line (black) with 3 channels (GRB)
            line = np.zeros((height, 5, 3), dtype=np.uint8)

            # Concatenate frames and line
            combined_frame = np.hstack((frame0, line, frame1))

            out.write(combined_frame)

    # Calculate time taken to capture the frame
    frame_duration = time.time() - frame_start_time

    # Sleep to maintain the exact frame rate
    sleep_time = frame_time - frame_duration
    if sleep_time > 0:
        time.sleep(sleep_time)

# Stop the cameras
cam0.stop()
cam1.stop()
