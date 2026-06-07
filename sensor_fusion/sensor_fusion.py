import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
from pyglet.window import key
from PIL import Image
import sys
import math
from DIPPID import SensorUDP

# --- Configuration ---
video_id = 0
if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

PORT = 5700
TRACK_ID_1 = 5
TRACK_ID_2 = 23 # Fallback if user uses phone screen
ACCEL_SCALAR = 2500.0 # Tuning multiplier for accelerometer

# --- Sensor Fusion State ---
camera_pos = None # Raw from ArUco (x, y)
pred_pos = [320.0, 240.0] # Fused position (x, y)
velocity = [0.0, 0.0]     # Velocity (vx, vy)
alpha = 0.5               # Weight for camera (0.0 to 1.0)

accel_data = {'x': 0.0, 'y': 0.0, 'z': 0.0}
reset_requested = False
has_dippid_data = False

# --- DIPPID Setup ---
sensor = SensorUDP(PORT)

def handle_accelerometer(data):
    global accel_data, has_dippid_data
    has_dippid_data = True
    if isinstance(data, dict):
        accel_data = data
    elif isinstance(data, str):
        import json
        try:
            accel_data = json.loads(data)
        except:
            pass

def handle_button_1(data):
    global reset_requested
    # DIPPID usually sends 1 for pressed, 0 for released
    if data == 1 or data == '1':
        print("[DEBUG] Button 1 pressed! Resetting prediction...")
        reset_requested = True

# Register callbacks
sensor.register_callback('accelerometer', handle_accelerometer)
# Try both common naming conventions for button 1 just in case
sensor.register_callback('button_1', handle_button_1)
try:
    sensor.register_callback('button1', handle_button_1)
except:
    pass

# --- OpenCV Setup ---
cap = cv2.VideoCapture(video_id)
cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if cam_width == 0 or cam_height == 0:
    cam_width, cam_height = 640, 480
    
pred_pos = [cam_width / 2.0, cam_height / 2.0]

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
    pyimg = pyglet.image.ImageData(width=cols, height=rows, fmt=fmt, data=raw_img, pitch=top_to_bottom_flag * bytes_per_row)
    return pyimg

# --- Pyglet Setup ---
window = pyglet.window.Window(cam_width, cam_height, caption="Sensor Fusion: ArUco + Accelerometer")
batch = pyglet.graphics.Batch()

alpha_label = pyglet.text.Label(f'Alpha (Up/Down): {alpha:.2f}', font_name='Arial', font_size=24,
                                x=20, y=cam_height - 30,
                                anchor_x='left', anchor_y='center',
                                color=(255, 255, 255, 255),
                                batch=batch)
                                
dippid_warning = pyglet.text.Label("WAITING FOR DIPPID ACCELEROMETER DATA ON PORT 5700", font_name='Arial', font_size=20,
                                x=cam_width//2, y=cam_height - 80,
                                anchor_x='center', anchor_y='center',
                                color=(255, 0, 0, 255))
                                
bg_sprite = None

@window.event
def on_key_press(symbol, modifiers):
    global alpha
    if symbol == key.UP:
        alpha = min(1.0, alpha + 0.05)
    elif symbol == key.DOWN:
        alpha = max(0.0, alpha - 0.05)
    alpha_label.text = f'Alpha (Up/Down): {alpha:.2f}'

def update(dt):
    global last_M, bg_sprite, camera_pos, pred_pos, velocity, reset_requested
    
    ret, frame = cap.read()
    if not ret: return
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    warped_frame = frame.copy()
    
    # Track the board bounds
    board_corners = []
    if ids is not None:
        for i in range(len(ids)):
            marker_id = ids[i][0]
            if marker_id != TRACK_ID_1 and marker_id != TRACK_ID_2:
                c = corners[i][0]
                board_corners.append([int(c[:, 0].mean()), int(c[:, 1].mean())])
                
    if len(board_corners) >= 4:
        src_pts = np.array(board_corners[:4], dtype=np.float32)
        src_pts = order_points(src_pts)
        dst_pts = np.array([[0, 0], [cam_width - 1, 0], [cam_width - 1, cam_height - 1], [0, cam_height - 1]], dtype=np.float32)
        last_M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
    if last_M is not None:
        warped_frame = cv2.warpPerspective(frame, last_M, (cam_width, cam_height))
        
        w_gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        w_corners, w_ids, _ = detector.detectMarkers(w_gray)
        
        camera_pos = None
        if w_ids is not None:
            for i in range(len(w_ids)):
                marker_id = w_ids[i][0]
                if marker_id == TRACK_ID_1 or marker_id == TRACK_ID_2:
                    c = w_corners[i][0]
                    cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
                    camera_pos = (cx, cy)
                    break
                    
    # --- Sensor Fusion ---
    if has_dippid_data:
        ax = float(accel_data.get('x', 0.0))
        ay = float(accel_data.get('y', 0.0))
        
        # Integrate acceleration into velocity. 
        velocity[0] += ax * ACCEL_SCALAR * dt
        velocity[1] += ay * ACCEL_SCALAR * dt
        
        # Friction
        velocity[0] *= 0.85
        velocity[1] *= 0.85

        # Accelerometer position prediction
        pred_pos[0] += velocity[0] * dt
        pred_pos[1] -= velocity[1] * dt # Y is usually inverted
    else:
        # If no DIPPID data, don't accumulate fake velocity
        velocity = [0.0, 0.0]

    # Complementary filter
    if camera_pos:
        if has_dippid_data:
            pred_pos[0] = alpha * camera_pos[0] + (1.0 - alpha) * pred_pos[0]
            pred_pos[1] = alpha * camera_pos[1] + (1.0 - alpha) * pred_pos[1]
        else:
            # If DIPPID isn't sending data, the "prediction" just snaps to the camera.
            # We will show the warning label so the user knows it's broken.
            pred_pos[0] = camera_pos[0]
            pred_pos[1] = camera_pos[1]
            
    # Boundary constraints
    pred_pos[0] = max(0, min(cam_width, pred_pos[0]))
    pred_pos[1] = max(0, min(cam_height, pred_pos[1]))

    # Reset mechanism
    if reset_requested:
        if camera_pos:
            pred_pos[0] = camera_pos[0]
            pred_pos[1] = camera_pos[1]
        else:
            pred_pos = [cam_width / 2.0, cam_height / 2.0]
        velocity = [0.0, 0.0]
        reset_requested = False

    if camera_pos:
        cv2.circle(warped_frame, (int(camera_pos[0]), int(camera_pos[1])), 15, (0, 0, 255), -1)

    if last_M is None:
        cv2.putText(warped_frame, "Searching for 4 ArUco Markers...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    rgb_frame = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2RGB)
    pyglet_img = cv2glet(rgb_frame, 'RGB')
    bg_sprite = pyglet.sprite.Sprite(pyglet_img, x=0, y=0)

@window.event
def on_draw():
    window.clear()
    if bg_sprite:
        bg_sprite.draw()
        
    # Draw Predicted Position (Green Dot) ONLY if we have DIPPID connection, 
    # OR draw it anyway but display a big warning if no data is coming in.
    pyglet_y = cam_height - pred_pos[1]
    pred_circle = pyglet.shapes.Circle(pred_pos[0], pyglet_y, 10, color=(0, 255, 0), batch=batch)
    
    batch.draw()
    
    if not has_dippid_data:
        dippid_warning.draw()

pyglet.clock.schedule_interval(update, 1/30.0)

if __name__ == "__main__":
    print("\n--- Sensor Fusion Started ---")
    print("Use UP/DOWN arrows to adjust Alpha.")
    print("Press Button 1 in DIPPID to reset the prediction position.")
    print("Close the window or press CTRL+C to exit.\n")
    try:
        pyglet.app.run()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sensor.disconnect()
