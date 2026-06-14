"""
Reflection on Implementation:
The complementary filter successfully blends the absolute position tracking from the camera (red dot) with the high-frequency relative motion data from the smartphone's accelerometer (fused into the green dot). 
When 'alpha' is high (closer to 1.0), the prediction relies heavily on the accelerometer data; this makes it very responsive to quick movements but susceptible to drift over time because accelerometer integration accumulates error. 
When 'alpha' is low (closer to 0.0), the prediction relies heavily on the camera; it becomes very stable and accurate in absolute space, but might feel slightly laggy due to the camera's framerate or processing delay.
By dynamically adjusting 'alpha', we can find the sweet spot where the prediction is both responsive (thanks to accelerometer) and stable (thanks to camera correction).
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
from pyglet.window import key
import sys
import math

from DIPPID import SensorUDP

# --- Configuration ---
PORT = 5700
if len(sys.argv) > 1:
    PORT = int(sys.argv[1])

print("Starting Sensor Fusion...")
print(__doc__)

# Initialize DIPPID
sensor = SensorUDP(PORT)

accel_data = {'x': 0.0, 'y': 0.0, 'z': 0.0}
reset_requested = False

def handle_accel(data):
    global accel_data
    accel_data = data

def handle_button_1(data):
    global reset_requested
    if data == 1:
        reset_requested = True

sensor.register_callback('accelerometer', handle_accel)
sensor.register_callback('button_1', handle_button_1)

# Pyglet window & Cam setup
cap = cv2.VideoCapture(0)
WINDOW_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
WINDOW_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if WINDOW_WIDTH == 0 or WINDOW_HEIGHT == 0:
    WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, caption="Sensor Fusion")

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# --- State ---
alpha = 0.5
camera_pos = [WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2]
pred_pos = [WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2]
ACCEL_SCALAR = 250.0  # Fixed scalar to convert accelerometer to pixels directly
gravity_filter = {'x': 0.0, 'y': 0.0}

board_marker_ids = set()

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def cv2glet(img, fmt='BGR'):
    if len(img.shape) == 2:
        rows, cols = img.shape
        channels = 1
        fmt = 'L'
    else:
        rows, cols, channels = img.shape
    raw_img = img.tobytes()
    top_to_bottom_flag = -1
    bytes_per_row = channels * cols
    pyimg = pyglet.image.ImageData(width=cols, height=rows, fmt=fmt, data=raw_img, pitch=top_to_bottom_flag * bytes_per_row)
    return pyimg

frame_to_draw = None

def update(dt):
    global frame_to_draw, camera_pos, pred_pos, reset_requested, alpha, board_marker_ids, gravity_filter

    ret, frame = cap.read()
    if not ret: return

    if frame.shape[1] != WINDOW_WIDTH or frame.shape[0] != WINDOW_HEIGHT:
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    
    warped_frame = None

    if ids is not None and len(ids) >= 4:
        # Assume the first 4 markers shown are the board if board_marker_ids is empty
        if len(board_marker_ids) < 4:
            board_marker_ids = set([ids[i][0] for i in range(4)])
            
        marker_centers = []
        for i in range(len(ids)):
            if ids[i][0] in board_marker_ids:
                c = corners[i][0]
                marker_centers.append([np.mean(c[:, 0]), np.mean(c[:, 1])])
                if len(marker_centers) == 4:
                    break
        
        if len(marker_centers) == 4:
            pts = np.array(marker_centers, dtype="float32")
            rect = order_points(pts)

            dst = np.array([
                [0, 0],
                [WINDOW_WIDTH - 1, 0],
                [WINDOW_WIDTH - 1, WINDOW_HEIGHT - 1],
                [0, WINDOW_HEIGHT - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped_frame = cv2.warpPerspective(frame, M, (WINDOW_WIDTH, WINDOW_HEIGHT))

            # Detect the moving marker in the warped frame
            warped_gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
            w_corners, w_ids, _ = detector.detectMarkers(warped_gray)

            if w_ids is not None:
                for i in range(len(w_ids)):
                    if w_ids[i][0] not in board_marker_ids:
                        # Found the moving marker (ID 5 or 23, or any non-board marker)
                        c = w_corners[i][0]
                        camera_pos[0] = WINDOW_WIDTH - np.mean(c[:, 0]) # Mirror X coordinate
                        camera_pos[1] = np.mean(c[:, 1])
                        break

    # Mirror the display for intuitive feedback
    if warped_frame is not None:
        display_frame = cv2.flip(warped_frame, 1)
    else:
        display_frame = cv2.flip(frame.copy(), 1)

    if reset_requested:
        pred_pos = [camera_pos[0], camera_pos[1]]
        reset_requested = False

    # Isolate gravity (Low-Pass Filter)
    gravity_filter['x'] = 0.98 * gravity_filter['x'] + 0.02 * accel_data['x']
    gravity_filter['y'] = 0.98 * gravity_filter['y'] + 0.02 * accel_data['y']
    
    # Extract linear acceleration (High-Pass Filter)
    lin_ax = accel_data['x'] - gravity_filter['x']
    lin_ay = accel_data['y'] - gravity_filter['y']

    # Assignment Hint: "multiply a fixed scalar number to the accelerometer"
    # This beautifully simple approach converts acceleration directly to a position offset!
    # Both axes receive a minus sign to match your specific phone's orientation and the mirrored display.
    offset_x = -lin_ax * ACCEL_SCALAR
    offset_y = lin_ay * ACCEL_SCALAR * 1.5  # Slightly increased Y sensitivity

    # Complementary filter: blend camera position with our accelerometer-offset prediction
    pred_pos[0] = alpha * (pred_pos[0] + offset_x) + (1 - alpha) * camera_pos[0]
    pred_pos[1] = alpha * (pred_pos[1] + offset_y) + (1 - alpha) * camera_pos[1]

    # Draw Camera Position (Red Dot)
    cv2.circle(display_frame, (int(camera_pos[0]), int(camera_pos[1])), 15, (0, 0, 255), -1)
    
    # Draw Predicted Position (Green Dot)
    cv2.circle(display_frame, (int(pred_pos[0]), int(pred_pos[1])), 10, (0, 255, 0), -1)

    # UI text
    cv2.putText(display_frame, f"Alpha: {alpha:.2f} (Use Left/Right Arrows)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, "DIPPID Btn 1 to Reset Prediction", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, "Red: Camera | Green: Prediction", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, "ESC/Q: Quit | F: Fullscreen", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display Accelerometer data so user knows it is connected
    accel_text = f"DIPPID Accel -> X: {accel_data['x']:.2f} | Y: {accel_data['y']:.2f}"
    cv2.putText(display_frame, accel_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if warped_frame is None:
        cv2.putText(display_frame, "Waiting for 4 ArUco Board Markers...", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    frame_to_draw = cv2glet(display_frame, 'BGR')

@window.event
def on_key_press(symbol, modifiers):
    global alpha
    if symbol == key.ESCAPE or symbol == key.Q:
        pyglet.app.exit()
    elif symbol == key.RIGHT:
        alpha = min(1.0, alpha + 0.05)
    elif symbol == key.LEFT:
        alpha = max(0.0, alpha - 0.05)
    elif symbol == key.F:
        window.set_fullscreen(not window.fullscreen)

@window.event
def on_draw():
    window.clear()
    if frame_to_draw:
        frame_to_draw.blit(0, 0, width=window.width, height=window.height)

# Schedule update at 30 fps
pyglet.clock.schedule_interval(update, 1/30.0)

if __name__ == "__main__":
    try:
        pyglet.app.run()
    finally:
        cap.release()
        sensor.disconnect()
