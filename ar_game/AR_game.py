import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
from PIL import Image
import sys
import random
import math

video_id = 0
if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# --- OpenCV Setup ---
cap = cv2.VideoCapture(video_id)
cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if cam_width == 0 or cam_height == 0:
    cam_width, cam_height = 640, 480

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

last_M = None

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def cv2glet(img, fmt):
    if fmt == 'GRAY':
        rows, cols = img.shape
        channels = 1
    else:
        rows, cols, channels = img.shape

    raw_img = Image.fromarray(img).tobytes()
    top_to_bottom_flag = -1
    bytes_per_row = channels * cols
    pyimg = pyglet.image.ImageData(width=cols, 
                                   height=rows, 
                                   fmt=fmt, 
                                   data=raw_img, 
                                   pitch=top_to_bottom_flag * bytes_per_row)
    return pyimg

# --- Game State: Simple Bubble Popper ---
score = 0
target_x = 0
target_y = 0
TARGET_RADIUS = 40
PLAYER_RADIUS = 15

def respawn_target():
    global target_x, target_y
    target_x = random.randint(TARGET_RADIUS * 2, cam_width - TARGET_RADIUS * 2)
    target_y = random.randint(TARGET_RADIUS * 2, cam_height - TARGET_RADIUS * 2)

# Spawn the very first target
respawn_target()

# --- Pyglet Setup ---
window = pyglet.window.Window(cam_width, cam_height, caption="AR Bubble Popper")
batch = pyglet.graphics.Batch()

score_label = pyglet.text.Label(f'Score: {score}', font_name='Arial', font_size=32,
                                x=20, y=cam_height - 40,
                                anchor_x='left', anchor_y='center',
                                color=(255, 255, 0, 255),
                                batch=batch)
                                
bg_sprite = None
player_pos = None

def update(dt):
    global last_M, bg_sprite, player_pos, score
    
    ret, frame = cap.read()
    if not ret: return
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    warped_frame = frame.copy()
    
    # Try to warp board
    if ids is not None and len(corners) >= 4:
        centers = []
        for i in range(4):
            c = corners[i][0]
            centers.append([int(c[:, 0].mean()), int(c[:, 1].mean())])
            
        src_pts = np.array(centers, dtype=np.float32)
        src_pts = order_points(src_pts)
        dst_pts = np.array([[0, 0], [cam_width - 1, 0], [cam_width - 1, cam_height - 1], [0, cam_height - 1]], dtype=np.float32)
        last_M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
    if last_M is not None:
        warped_frame = cv2.warpPerspective(frame, last_M, (cam_width, cam_height))
        
    # Object tracking (Red)
    hsv = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    mask = mask1 + mask2
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    player_pos = None
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                player_pos = (cx, cy)
                
                # Draw player circle
                cv2.circle(warped_frame, (cx, cy), PLAYER_RADIUS, (0, 255, 0), -1)

    # Game Logic: Check collision with the single target
    if player_pos:
        px, py = player_pos
        distance = math.hypot(px - target_x, py - target_y)
        
        if distance < TARGET_RADIUS + PLAYER_RADIUS:
            # Touched the target!
            score += 1
            score_label.text = f'Score: {score}'
            respawn_target()

    # Visual debug for ArUco
    if last_M is None:
        cv2.putText(warped_frame, "Searching for 4 ArUco Markers...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Update Pyglet Background
    rgb_frame = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2RGB)
    pyglet_img = cv2glet(rgb_frame, 'RGB')
    bg_sprite = pyglet.sprite.Sprite(pyglet_img, x=0, y=0)

@window.event
def on_draw():
    window.clear()
    if bg_sprite:
        bg_sprite.draw()
        
    # Draw the single target
    shapes = []
    pyglet_y = cam_height - target_y
    target_circle = pyglet.shapes.Circle(target_x, pyglet_y, TARGET_RADIUS, color=(50, 150, 255), batch=batch)
    shapes.append(target_circle)
            
    batch.draw()

pyglet.clock.schedule_interval(update, 1/30.0)

if __name__ == "__main__":
    pyglet.app.run()
    cap.release()
    cv2.destroyAllWindows()
