import cv2
from flask import Flask, Response
from picamera2 import Picamera2
import time

width = 1280
height = 720

app = Flask(__name__)


def generate_frames_0():
    while True:
        frame = cam0.capture_array()
        if frame is None:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Show the live frames using OpenCV
            # cv2.imshow('Live Feed', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def generate_frames_1():
    while True:
        frame = cam1.capture_array()
        if frame is None:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Show the live frames using OpenCV
            # cv2.imshow('Live Feed', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed_0')
def video_feed_0():
    return Response(generate_frames_0(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_1')
def video_feed_1():
    return Response(generate_frames_1(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return "<html><body><img src='/video_feed_0' width='640'><img src='/video_feed_1' width='640'></body></html>"


if __name__ == '__main__':
    cam0 = Picamera2(0)
    cam0.configure(cam0.create_video_configuration({'size': (width, height), 'format': 'RGB888'}))
    cam0.start()
    time.sleep(2)

    cam1 = Picamera2(1)
    cam1.configure(cam1.create_video_configuration({'size': (width, height), 'format': 'RGB888'}))
    cam1.start()
    time.sleep(2)

    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    cam0.stop()
    cam1.stop()
