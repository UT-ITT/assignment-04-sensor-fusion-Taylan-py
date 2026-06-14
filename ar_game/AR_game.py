import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
from pyglet.window import key
import sys
import random

# Parse arguments for video source
video_source = 0
if len(sys.argv) > 1:
    try:
        video_source = int(sys.argv[1])
    except ValueError:
        video_source = sys.argv[1]

# Initialize video capture
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print(f"Error: Could not open video source {video_source}")
    sys.exit(1)

# Get resolution
WINDOW_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
WINDOW_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if WINDOW_WIDTH == 0 or WINDOW_HEIGHT == 0:
    WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, caption="AR Target Destroyer")

# ArUco parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# Game State
score = 0
target_pos = None
target_radius = 40

def spawn_target():
    global target_pos
    margin = target_radius + 10
    x = random.randint(margin, WINDOW_WIDTH - margin)
    y = random.randint(margin, WINDOW_HEIGHT - margin)
    target_pos = (x, y)

spawn_target()

def cv2glet(img, fmt='BGR'):
    '''Converts OpenCV image to pyglet image object'''
    if len(img.shape) == 2:
        rows, cols = img.shape
        channels = 1
        fmt = 'L'
    else:
        rows, cols, channels = img.shape

    raw_img = img.tobytes()

    top_to_bottom_flag = -1
    bytes_per_row = channels * cols
    pyimg = pyglet.image.ImageData(width=cols, 
                                   height=rows, 
                                   fmt=fmt, 
                                   data=raw_img, 
                                   pitch=top_to_bottom_flag * bytes_per_row)
    return pyimg

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

frame_to_draw = None

def update(dt):
    global frame_to_draw, score, target_pos
    
    ret, frame = cap.read()
    if not ret:
        # Loop video if it's a file, or ignore if it's live feed issue
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            return

    # Resize frame if necessary
    if frame.shape[1] != WINDOW_WIDTH or frame.shape[0] != WINDOW_HEIGHT:
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    
    warped_frame = None

    if ids is not None and len(ids) >= 4:
        # Take the first 4 markers found
        marker_centers = []
        for i in range(4):
            c = corners[i][0]
            center_x = np.mean(c[:, 0])
            center_y = np.mean(c[:, 1])
            marker_centers.append([center_x, center_y])
        
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
    
    display_frame = warped_frame if warped_frame is not None else frame.copy()

    if warped_frame is not None:
        # Object tracking (Blue Color) for the player
        hsv = cv2.cvtColor(display_frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 100, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Clean up mask noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tracked_pos = None

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 300: # Minimum size to avoid noise
                M_moments = cv2.moments(c)
                if M_moments["m00"] != 0:
                    cX = int(M_moments["m10"] / M_moments["m00"])
                    cY = int(M_moments["m01"] / M_moments["m00"])
                    tracked_pos = (cX, cY)
                    
                    # Draw tracker crosshair
                    cv2.circle(display_frame, tracked_pos, 10, (0, 255, 0), -1)
                    cv2.putText(display_frame, "Player", (cX - 25, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Game Logic: check for collision with target
        if tracked_pos and target_pos:
            dist = np.sqrt((tracked_pos[0] - target_pos[0])**2 + (tracked_pos[1] - target_pos[1])**2)
            if dist < target_radius + 10: # collision
                score += 1
                spawn_target()
        
        # Draw target (Looks like a bullseye)
        if target_pos:
            cv2.circle(display_frame, target_pos, target_radius, (0, 0, 255), -1)
            cv2.circle(display_frame, target_pos, int(target_radius*0.6), (255, 255, 255), -1)
            cv2.circle(display_frame, target_pos, int(target_radius*0.3), (0, 0, 255), -1)
            
        # Draw Score
        cv2.putText(display_frame, f"Score: {score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        # Fallback text when 4 markers aren't found
        text = "Show ArUco Board (4 markers) to Play!"
        cv2.putText(display_frame, text, (22, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(display_frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        text2 = "Hold a BLUE object (e.g., phone screen) to play."
        cv2.putText(display_frame, text2, (22, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(display_frame, text2, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Draw legend
    legend_text = "ESC / Q: Quit"
    cv2.putText(display_frame, legend_text, (22, WINDOW_HEIGHT - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(display_frame, legend_text, (20, WINDOW_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    frame_to_draw = cv2glet(display_frame, 'BGR')

@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.ESCAPE or symbol == key.Q:
        pyglet.app.exit()

@window.event
def on_draw():
    window.clear()
    if frame_to_draw:
        frame_to_draw.blit(0, 0, 0)

# Schedule update at 30 fps
pyglet.clock.schedule_interval(update, 1/30.0)

if __name__ == "__main__":
    try:
        pyglet.app.run()
    finally:
        cap.release()
