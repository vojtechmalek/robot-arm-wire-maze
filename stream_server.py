from flask import Flask, Response, render_template_string
import threading
import cv2
import numpy as np
import time

try:
    from ctu_crs import CRS97  # or CRS93
except Exception:
    CRS97 = None

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<title>Robot Camera Stream</title>
<h1>Robot Live Stream</h1>
<img src="/video_feed" width="640" />
<p>Press Ctrl-C in the terminal to stop the server.</p>
"""


class RobotCamera:
    """Simple wrapper that grabs images from the robot and encodes to JPEG.

    If `ctu_crs` is not available or creation fails, images will be a black frame.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.robot = None
        self.running = False
        # try to create robot
        if CRS97 is not None:
            try:
                self.robot = CRS97()
                # do a non-blocking initialize where possible
                try:
                    self.robot.initialize(home=False)
                except Exception:
                    # ignore initialization errors; robot may already be initialized
                    pass
            except Exception:
                self.robot = None

        # placeholder frame (black) in case real camera isn't available
        self.placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        self.running = True

    def get_raw_image(self):
        """Return an RGB image (numpy array) or placeholder on failure."""
        if not self.running:
            return None
        if self.robot is None:
            return self.placeholder
        try:
            img = self.robot.grab_image()
            if img is None or not isinstance(img, np.ndarray):
                return self.placeholder
            return img
        except Exception:
            return self.placeholder

    def get_frame(self, scale=0.5, quality=80):
        """Return JPEG-encoded bytes for the current frame."""
        img = self.get_raw_image()
        if img is None:
            return None
        try:
            # optionally scale for faster streaming
            if scale != 1.0:
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

            # ensure BGR -> JPEG works
            ret, jpeg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ret:
                return None
            return jpeg.tobytes()
        except Exception:
            return None

    def close(self):
        self.running = False
        try:
            if self.robot is not None:
                self.robot.close()
        except Exception:
            pass


camera = RobotCamera()


def gen(camera):
    """Video streaming generator (MJPEG)."""
    boundary = b'--frame'
    while camera.running:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # small sleep to yield CPU and limit frame rate
        time.sleep(0.02)


@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


@app.route('/video_feed')
def video_feed():
    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


def run_server(host='0.0.0.0', port=5000):
    try:
        # Flask's built-in server is fine for local networks and development.
        app.run(host=host, port=port, threaded=True)
    finally:
        camera.close()


if __name__ == '__main__':
    try:
        print('Starting video server on http://0.0.0.0:5000')
        run_server()
    except KeyboardInterrupt:
        print('Shutting down...')
        camera.close()
