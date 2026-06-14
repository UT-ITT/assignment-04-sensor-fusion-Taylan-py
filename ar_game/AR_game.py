import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
from pyglet.window import key
import sys
import random
import time

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

# Get resolution from webcam to check if it's working
cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if cam_w == 0 or cam_h == 0:
    print("Warning: Could not read webcam resolution")

# Create windowed by default (press 'F' to toggle fullscreen)
WINDOW_WIDTH = cam_w
WINDOW_HEIGHT = cam_h
if WINDOW_WIDTH == 0 or WINDOW_HEIGHT == 0:
    WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, caption="AR Target Destroyer")

# ArUco parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# Game State
score = 0
target_radius = 40
TOTAL_TARGETS = 10
game_state = "PLAYING" # PLAYING, GAME_OVER
start_time = time.time()
best_time = None

# Maze State
CELL_SIZE = 80
maze_walls = []
start_zone = None
target_zone = None
player_state = "ALIVE"

PADDING = 120 # Padding to keep maze away from ArUco markers

def generate_maze():
    global maze_walls, start_zone, target_zone, player_state
    w = max(640, WINDOW_WIDTH) - 2 * PADDING
    h = max(480, WINDOW_HEIGHT) - 2 * PADDING
    c = max(2, w // CELL_SIZE)
    r = max(2, h // CELL_SIZE)
    
    grid = [[{'visited': False, 'walls': {'N': True, 'E': True, 'S': True, 'W': True}} for _ in range(c)] for _ in range(r)]
    
    def visit(x, y):
        grid[y][x]['visited'] = True
        directions = [('N', 0, -1, 'S'), ('S', 0, 1, 'N'), ('E', 1, 0, 'W'), ('W', -1, 0, 'E')]
        random.shuffle(directions)
        for dir_name, dx, dy, opposite in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < c and 0 <= ny < r and not grid[ny][nx]['visited']:
                grid[y][x]['walls'][dir_name] = False
                grid[ny][nx]['walls'][opposite] = False
                visit(nx, ny)
                
    # Increase recursion limit just in case
    sys.setrecursionlimit(2000)
    visit(0, 0)
    
    maze_walls = []
    # Calculate offset to center the maze if there's remainder space
    offset_x = PADDING + (w - c * CELL_SIZE) // 2
    offset_y = PADDING + (h - r * CELL_SIZE) // 2
    
    for y in range(r):
        for x in range(c):
            px, py = offset_x + x * CELL_SIZE, offset_y + y * CELL_SIZE
            if grid[y][x]['walls']['N']:
                maze_walls.append(((px, py), (px + CELL_SIZE, py)))
            if grid[y][x]['walls']['E']:
                maze_walls.append(((px + CELL_SIZE, py), (px + CELL_SIZE, py + CELL_SIZE)))
            if grid[y][x]['walls']['S']:
                maze_walls.append(((px, py + CELL_SIZE), (px + CELL_SIZE, py + CELL_SIZE)))
            if grid[y][x]['walls']['W']:
                maze_walls.append(((px, py), (px, py + CELL_SIZE)))
                
    start_zone = (offset_x + CELL_SIZE // 2, offset_y + CELL_SIZE // 2)
    target_zone = (offset_x + (c - 1) * CELL_SIZE + CELL_SIZE // 2, offset_y + (r - 1) * CELL_SIZE + CELL_SIZE // 2)
    player_state = "ALIVE"

def get_wall_mask(w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    for start_pt, end_pt in maze_walls:
        cv2.line(mask, start_pt, end_pt, 255, 10)
    return mask

def reset_game():
    global score, game_state, start_time
    score = 0
    game_state = "PLAYING"
    start_time = time.time()
    generate_maze()

generate_maze()

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
    global frame_to_draw, score, target_zone, game_state, best_time, start_time, player_state, WINDOW_WIDTH, WINDOW_HEIGHT
    
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

    # Detect ArUco on the UN-FLIPPED frame! (Flipping breaks marker detection)
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
    
    # Now flip the frame for intuitive mirrored interaction and display
    if warped_frame is not None:
        display_frame = cv2.flip(warped_frame, 1)
    else:
        display_frame = cv2.flip(frame.copy(), 1)

    if warped_frame is not None:
        # Object tracking (Finger) using pure OpenCV Skin Color Tracking
        # Using YCrCb color space is much more robust to lighting changes than HSV!
        ycrcb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2YCrCb)
        
        # Define typical skin color range in YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tracked_pos = None
        
        if contours:
            # Find the largest contour (assuming it's the hand)
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 1000:
                # Find the topmost point of the contour (the fingertip when hand is held up!)
                extTop = tuple(c[c[:, :, 1].argmin()][0])
                tracked_pos = (int(extTop[0]), int(extTop[1]))
                cv2.drawContours(display_frame, [c], -1, (255, 0, 0), 2)

        if game_state == "PLAYING":
            current_time = time.time() - start_time
            
            # Draw Maze
            for start_pt, end_pt in maze_walls:
                cv2.line(display_frame, start_pt, end_pt, (0, 0, 0), 10)
            
            # Draw Start Zone
            cv2.circle(display_frame, start_zone, CELL_SIZE//3, (0, 255, 0), 3)
            cv2.putText(display_frame, "START", (start_zone[0]-25, start_zone[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw Target Zone
            cv2.circle(display_frame, target_zone, CELL_SIZE//3, (0, 0, 255), -1)
            cv2.circle(display_frame, target_zone, int((CELL_SIZE//3)*0.6), (255, 255, 255), -1)
            cv2.circle(display_frame, target_zone, int((CELL_SIZE//3)*0.3), (0, 0, 255), -1)
            
            # Game Logic
            if tracked_pos:
                tx, ty = tracked_pos
                tx = max(0, min(WINDOW_WIDTH-1, tx))
                ty = max(0, min(WINDOW_HEIGHT-1, ty))
                
                # Wall collision
                wall_mask = get_wall_mask(WINDOW_WIDTH, WINDOW_HEIGHT)
                if wall_mask[ty, tx] == 255:
                    player_state = "DEAD"
                    
                if player_state == "DEAD":
                    # Check if player returned to start
                    dist_start = np.sqrt((tx - start_zone[0])**2 + (ty - start_zone[1])**2)
                    if dist_start < CELL_SIZE//3:
                        player_state = "ALIVE"
                
                if player_state == "ALIVE":
                    # Check target collision
                    dist_target = np.sqrt((tx - target_zone[0])**2 + (ty - target_zone[1])**2)
                    if dist_target < CELL_SIZE//3:
                        score += 1
                        if score >= TOTAL_TARGETS:
                            game_state = "GAME_OVER"
                            if best_time is None or current_time < best_time:
                                best_time = current_time
                        else:
                            generate_maze()
            
            # Draw Player Tracker
            if tracked_pos:
                tx, ty = tracked_pos
                player_color = (0, 255, 0) if player_state == "ALIVE" else (0, 0, 255)
                cv2.circle(display_frame, tracked_pos, 10, player_color, -1)
                if player_state == "DEAD":
                    cv2.putText(display_frame, "HIT WALL! GO TO START!", (tx - 80, ty - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "Player", (tx - 25, ty - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, player_color, 2)
                
            # Draw Score & Timer
            # Add solid background to text so it's readable over maze
            cv2.rectangle(display_frame, (10, 10), (300, 130), (0,0,0), -1)
            cv2.putText(display_frame, f"Level: {score+1}/{TOTAL_TARGETS}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.putText(display_frame, f"Time: {current_time:.2f}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            if best_time is not None:
                cv2.putText(display_frame, f"Best: {best_time:.2f}s", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                
        elif game_state == "GAME_OVER":
            cv2.putText(display_frame, "YOU WIN!", (WINDOW_WIDTH//2 - 100, WINDOW_HEIGHT//2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
            cv2.putText(display_frame, "Press 'R' to Restart", (WINDOW_WIDTH//2 - 150, WINDOW_HEIGHT//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Best: {best_time:.2f}s", (WINDOW_WIDTH//2 - 100, WINDOW_HEIGHT//2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    else:
        # Fallback text when 4 markers aren't found
        text = "Show ArUco Board (4 markers) to Play!"
        cv2.putText(display_frame, text, (22, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(display_frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        text2 = "Hold your hand up to play."
        cv2.putText(display_frame, text2, (22, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(display_frame, text2, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Draw legend
    legend_text = "ESC / Q: Quit | F: Fullscreen"
    cv2.putText(display_frame, legend_text, (22, WINDOW_HEIGHT - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(display_frame, legend_text, (20, WINDOW_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    frame_to_draw = cv2glet(display_frame, 'BGR')

@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.ESCAPE or symbol == key.Q:
        pyglet.app.exit()
    elif symbol == key.R:
        if game_state == "GAME_OVER":
            reset_game()
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
